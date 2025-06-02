/*!
# Distributed Computing Module

This module provides MPI-based distributed computing capabilities for OpenAlgebra,
enabling large-scale sparse linear algebra operations across multiple nodes.
*/

use crate::{
    sparse::{CSRMatrix, SparseMatrix},
    solvers::{IterativeSolver, SolverInfo},
    Result, OpenAlgebraError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "mpi")]
use mpi::{
    collective::{CommunicatorCollectives, Root},
    datatype::{Partition, PartitionMut},
    environment::Universe,
    point_to_point::{Destination, Source},
    request::WaitGuard,
    topology::{Communicator, Rank, SystemCommunicator},
    traits::{Equivalence, Root as RootTrait},
};

/// MPI communicator wrapper for distributed operations
pub struct DistributedContext {
    #[cfg(feature = "mpi")]
    universe: Universe,
    #[cfg(feature = "mpi")]
    world: SystemCommunicator,
    rank: i32,
    size: i32,
}

/// Distributed sparse matrix representation
#[derive(Debug, Clone)]
pub struct DistributedCSRMatrix<T> {
    local_matrix: CSRMatrix<T>,
    global_rows: usize,
    global_cols: usize,
    row_distribution: Vec<usize>, // Number of rows per process
    col_distribution: Vec<usize>, // Number of columns per process
    row_offset: usize,            // Starting row index for this process
    col_offset: usize,            // Starting column index for this process
}

/// Distributed vector representation
#[derive(Debug, Clone)]
pub struct DistributedVector<T> {
    local_data: Vec<T>,
    global_size: usize,
    distribution: Vec<usize>, // Elements per process
    offset: usize,            // Starting index for this process
}

/// Communication pattern for distributed operations
#[derive(Debug, Clone)]
pub struct CommunicationPattern {
    send_counts: Vec<usize>,
    recv_counts: Vec<usize>,
    send_displs: Vec<usize>,
    recv_displs: Vec<usize>,
}

/// Distributed solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSolverConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub restart: Option<usize>, // For GMRES
    pub overlap_communication: bool,
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RowWise,
    BlockCyclic { block_size: usize },
    Graph { partitioner: String },
    Custom(Vec<usize>),
}

impl DistributedContext {
    /// Initialize MPI context
    pub fn new() -> Result<Self> {
        #[cfg(feature = "mpi")]
        {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let rank = world.rank();
            let size = world.size();

            Ok(Self {
                universe,
                world,
                rank,
                size,
            })
        }

        #[cfg(not(feature = "mpi"))]
        {
            Ok(Self {
                rank: 0,
                size: 1,
            })
        }
    }

    /// Get process rank
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get total number of processes
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            self.world.barrier();
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            Ok(())
        }
    }

    /// Broadcast data from root to all processes
    pub fn broadcast<T>(&self, data: &mut [T], root: i32) -> Result<()>
    where
        T: Equivalence,
    {
        #[cfg(feature = "mpi")]
        {
            if self.rank == root {
                self.world.process_at_rank(root).broadcast_into(data);
            } else {
                self.world.process_at_rank(root).broadcast_into(data);
            }
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            Ok(())
        }
    }

    /// All-reduce operation
    pub fn allreduce<T>(&self, send_data: &[T], recv_data: &mut [T]) -> Result<()>
    where
        T: Equivalence + std::ops::Add<Output = T> + Copy,
    {
        #[cfg(feature = "mpi")]
        {
            self.world
                .all_reduce_into(send_data, recv_data, &mpi::collective::SystemOperation::sum());
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            recv_data.copy_from_slice(send_data);
            Ok(())
        }
    }

    /// Gather data from all processes to root
    pub fn gather<T>(&self, send_data: &[T], recv_data: &mut [T], root: i32) -> Result<()>
    where
        T: Equivalence,
    {
        #[cfg(feature = "mpi")]
        {
            if self.rank == root {
                self.world
                    .process_at_rank(root)
                    .gather_into_root(send_data, recv_data);
            } else {
                self.world
                    .process_at_rank(root)
                    .gather_into(send_data);
            }
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            recv_data[..send_data.len()].copy_from_slice(send_data);
            Ok(())
        }
    }

    /// Scatter data from root to all processes
    pub fn scatter<T>(&self, send_data: &[T], recv_data: &mut [T], root: i32) -> Result<()>
    where
        T: Equivalence,
    {
        #[cfg(feature = "mpi")]
        {
            if self.rank == root {
                self.world
                    .process_at_rank(root)
                    .scatter_into_root(send_data, recv_data);
            } else {
                self.world
                    .process_at_rank(root)
                    .scatter_into(recv_data);
            }
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            recv_data.copy_from_slice(&send_data[..recv_data.len()]);
            Ok(())
        }
    }

    /// All-gather operation
    pub fn allgather<T>(&self, send_data: &[T], recv_data: &mut [T]) -> Result<()>
    where
        T: Equivalence,
    {
        #[cfg(feature = "mpi")]
        {
            self.world.all_gather_into(send_data, recv_data);
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            recv_data[..send_data.len()].copy_from_slice(send_data);
            Ok(())
        }
    }
}

impl<T> DistributedCSRMatrix<T>
where
    T: Clone + Default + Copy,
{
    /// Create distributed matrix from local CSR matrix
    pub fn new(
        local_matrix: CSRMatrix<T>,
        global_rows: usize,
        global_cols: usize,
        row_distribution: Vec<usize>,
        col_distribution: Vec<usize>,
        row_offset: usize,
        col_offset: usize,
    ) -> Self {
        Self {
            local_matrix,
            global_rows,
            global_cols,
            row_distribution,
            col_distribution,
            row_offset,
            col_offset,
        }
    }

    /// Distribute matrix using row-wise partitioning
    pub fn from_csr_rowwise(
        matrix: &CSRMatrix<T>,
        context: &DistributedContext,
    ) -> Result<Self> {
        let rows_per_proc = matrix.rows() / context.size() as usize;
        let remainder = matrix.rows() % context.size() as usize;

        let mut row_distribution = vec![rows_per_proc; context.size() as usize];
        for i in 0..remainder {
            row_distribution[i] += 1;
        }

        let row_offset = row_distribution[..context.rank() as usize]
            .iter()
            .sum::<usize>();
        let local_rows = row_distribution[context.rank() as usize];

        // Extract local portion of the matrix
        let start_row = row_offset;
        let end_row = start_row + local_rows;

        let local_matrix = matrix.submatrix_rows(start_row, end_row)?;

        Ok(Self::new(
            local_matrix,
            matrix.rows(),
            matrix.cols(),
            row_distribution,
            vec![matrix.cols()], // Single column block
            row_offset,
            0,
        ))
    }

    /// Perform distributed matrix-vector multiplication
    pub fn matvec(
        &self,
        x: &DistributedVector<T>,
        y: &mut DistributedVector<T>,
        context: &DistributedContext,
    ) -> Result<()>
    where
        T: num_traits::Float + std::ops::AddAssign + Equivalence,
    {
        // Local matrix-vector multiplication
        let local_y = self.local_matrix.matvec(&x.local_data)?;
        y.local_data.copy_from_slice(&local_y);

        // Handle off-diagonal communication if needed
        if self.col_distribution.len() > 1 {
            // This would require more complex communication patterns
            // for matrices distributed in both rows and columns
            todo!("Column-distributed matrices not yet implemented");
        }

        Ok(())
    }

    /// Get local matrix
    pub fn local_matrix(&self) -> &CSRMatrix<T> {
        &self.local_matrix
    }

    /// Get global dimensions
    pub fn global_dimensions(&self) -> (usize, usize) {
        (self.global_rows, self.global_cols)
    }
}

impl<T> DistributedVector<T>
where
    T: Clone + Default + Copy,
{
    /// Create distributed vector
    pub fn new(
        local_data: Vec<T>,
        global_size: usize,
        distribution: Vec<usize>,
        offset: usize,
    ) -> Self {
        Self {
            local_data,
            global_size,
            distribution,
            offset,
        }
    }

    /// Create distributed vector with uniform distribution
    pub fn zeros_uniform(global_size: usize, context: &DistributedContext) -> Self {
        let elements_per_proc = global_size / context.size() as usize;
        let remainder = global_size % context.size() as usize;

        let mut distribution = vec![elements_per_proc; context.size() as usize];
        for i in 0..remainder {
            distribution[i] += 1;
        }

        let offset = distribution[..context.rank() as usize]
            .iter()
            .sum::<usize>();
        let local_size = distribution[context.rank() as usize];

        Self::new(
            vec![T::default(); local_size],
            global_size,
            distribution,
            offset,
        )
    }

    /// Compute distributed dot product
    pub fn dot(&self, other: &Self, context: &DistributedContext) -> Result<T>
    where
        T: num_traits::Float + Equivalence + std::ops::AddAssign,
    {
        // Local dot product
        let local_dot = self
            .local_data
            .iter()
            .zip(other.local_data.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x);

        // Global reduction
        let mut global_dot = T::zero();
        context.allreduce(&[local_dot], &mut [global_dot])?;

        Ok(global_dot)
    }

    /// Compute distributed norm
    pub fn norm(&self, context: &DistributedContext) -> Result<T>
    where
        T: num_traits::Float + Equivalence + std::ops::AddAssign,
    {
        let dot_product = self.dot(self, context)?;
        Ok(dot_product.sqrt())
    }

    /// Gather vector to root process
    pub fn gather(&self, context: &DistributedContext, root: i32) -> Result<Option<Vec<T>>>
    where
        T: Equivalence,
    {
        if context.rank() == root {
            let mut global_data = vec![T::default(); self.global_size];
            context.gather(&self.local_data, &mut global_data, root)?;
            Ok(Some(global_data))
        } else {
            context.gather(&self.local_data, &mut [], root)?;
            Ok(None)
        }
    }
}

/// Distributed Conjugate Gradient solver
pub struct DistributedConjugateGradient {
    config: DistributedSolverConfig,
}

impl DistributedConjugateGradient {
    /// Create new distributed CG solver
    pub fn new(config: DistributedSolverConfig) -> Self {
        Self { config }
    }

    /// Solve distributed linear system
    pub fn solve<T>(
        &self,
        matrix: &DistributedCSRMatrix<T>,
        b: &DistributedVector<T>,
        x: &mut DistributedVector<T>,
        context: &DistributedContext,
    ) -> Result<SolverInfo>
    where
        T: num_traits::Float + Equivalence + std::ops::AddAssign + std::fmt::Debug,
    {
        let mut iterations = 0;
        let mut residual_norm = T::one();

        // Initialize residual r = b - Ax
        let mut r = b.clone();
        let mut ax = DistributedVector::zeros_uniform(b.global_size, context);
        matrix.matvec(x, &mut ax, context)?;

        for i in 0..r.local_data.len() {
            r.local_data[i] = r.local_data[i] - ax.local_data[i];
        }

        let mut p = r.clone();
        let mut rsold = r.dot(&r, context)?;

        while residual_norm > T::from(self.config.tolerance).unwrap()
            && iterations < self.config.max_iterations
        {
            // Ap = A * p
            let mut ap = DistributedVector::zeros_uniform(b.global_size, context);
            matrix.matvec(&p, &mut ap, context)?;

            // alpha = rsold / (p^T * Ap)
            let pap = p.dot(&ap, context)?;
            let alpha = rsold / pap;

            // x = x + alpha * p
            for i in 0..x.local_data.len() {
                x.local_data[i] = x.local_data[i] + alpha * p.local_data[i];
            }

            // r = r - alpha * Ap
            for i in 0..r.local_data.len() {
                r.local_data[i] = r.local_data[i] - alpha * ap.local_data[i];
            }

            let rsnew = r.dot(&r, context)?;
            residual_norm = rsnew.sqrt();

            if residual_norm <= T::from(self.config.tolerance).unwrap() {
                break;
            }

            // beta = rsnew / rsold
            let beta = rsnew / rsold;

            // p = r + beta * p
            for i in 0..p.local_data.len() {
                p.local_data[i] = r.local_data[i] + beta * p.local_data[i];
            }

            rsold = rsnew;
            iterations += 1;
        }

        Ok(SolverInfo {
            converged: residual_norm <= T::from(self.config.tolerance).unwrap(),
            iterations,
            residual_norm: residual_norm.to_f64().unwrap_or(0.0),
            solve_time: 0.0, // Would measure actual time
        })
    }
}

/// Initialize MPI subsystem
pub fn init_mpi() -> Result<()> {
    #[cfg(feature = "mpi")]
    {
        println!("Initializing MPI distributed computing...");
        // MPI initialization is handled in DistributedContext::new()
        Ok(())
    }

    #[cfg(not(feature = "mpi"))]
    {
        println!("MPI support not enabled");
        Ok(())
    }
}

/// Utility functions for matrix distribution
pub mod distribution {
    use super::*;

    /// Calculate row distribution for load balancing
    pub fn calculate_row_distribution(
        total_rows: usize,
        num_processes: usize,
        strategy: &LoadBalancingStrategy,
    ) -> Vec<usize> {
        match strategy {
            LoadBalancingStrategy::RowWise => {
                let rows_per_proc = total_rows / num_processes;
                let remainder = total_rows % num_processes;
                let mut distribution = vec![rows_per_proc; num_processes];
                for i in 0..remainder {
                    distribution[i] += 1;
                }
                distribution
            }
            LoadBalancingStrategy::BlockCyclic { block_size } => {
                let mut distribution = vec![0; num_processes];
                let mut current_proc = 0;
                let mut remaining_rows = total_rows;

                while remaining_rows > 0 {
                    let chunk_size = (*block_size).min(remaining_rows);
                    distribution[current_proc] += chunk_size;
                    remaining_rows -= chunk_size;
                    current_proc = (current_proc + 1) % num_processes;
                }
                distribution
            }
            LoadBalancingStrategy::Custom(custom_dist) => custom_dist.clone(),
            LoadBalancingStrategy::Graph { .. } => {
                // Would implement graph partitioning algorithms
                // For now, fall back to row-wise
                calculate_row_distribution(total_rows, num_processes, &LoadBalancingStrategy::RowWise)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_context_creation() {
        let context = DistributedContext::new();
        assert!(context.is_ok());
        let ctx = context.unwrap();
        assert_eq!(ctx.rank(), 0); // Single process test
        assert_eq!(ctx.size(), 1);
    }

    #[test]
    fn test_row_distribution() {
        let dist = distribution::calculate_row_distribution(
            100,
            4,
            &LoadBalancingStrategy::RowWise,
        );
        assert_eq!(dist, vec![25, 25, 25, 25]);

        let dist = distribution::calculate_row_distribution(
            101,
            4,
            &LoadBalancingStrategy::RowWise,
        );
        assert_eq!(dist, vec![26, 25, 25, 25]);
    }

    #[test]
    fn test_block_cyclic_distribution() {
        let dist = distribution::calculate_row_distribution(
            100,
            4,
            &LoadBalancingStrategy::BlockCyclic { block_size: 10 },
        );
        // Each process should get 2.5 blocks = 25 rows
        assert_eq!(dist.iter().sum::<usize>(), 100);
        assert_eq!(dist.len(), 4);
    }

    #[test]
    fn test_distributed_vector_creation() {
        let context = DistributedContext::new().unwrap();
        let vec = DistributedVector::<f64>::zeros_uniform(100, &context);
        assert_eq!(vec.global_size, 100);
        assert_eq!(vec.local_data.len(), 100); // Single process
    }
} 