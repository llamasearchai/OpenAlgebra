/*!
# OpenAI Agents Integration for OpenAlgebra

This module provides integration with OpenAI's API for intelligent mathematical
problem solving and optimization using the OpenAlgebra library.
*/

use async_openai::{
    Client,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, CreateChatCompletionRequest,
        CreateChatCompletionResponse, Role,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use crate::{
    sparse::{COOMatrix, CSRMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver},
    tensor::{SparseTensor, DenseTensor, Tensor},
    preconditioners::{ILUPreconditioner, JacobiPreconditioner, Preconditioner},
    Result, OpenAlgebraError,
};

/// Configuration for OpenAI agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u16>,
    pub temperature: Option<f32>,
    pub timeout_seconds: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "gpt-4".to_string(),
            max_tokens: Some(2048),
            temperature: Some(0.7),
            timeout_seconds: 30,
        }
    }
}

/// Mathematical problem types that can be solved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProblemType {
    LinearSystem,
    EigenvalueProblem,
    Optimization,
    MatrixFactorization,
    TensorDecomposition,
    Custom(String),
}

/// Problem description for AI agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathProblem {
    pub problem_type: ProblemType,
    pub description: String,
    pub constraints: Vec<String>,
    pub objectives: Vec<String>,
    pub matrix_size: Option<(usize, usize)>,
    pub sparsity_pattern: Option<String>,
    pub numerical_properties: HashMap<String, f64>,
}

/// Solution strategy recommended by AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionStrategy {
    pub solver_type: String,
    pub preconditioner: Option<String>,
    pub parameters: HashMap<String, f64>,
    pub expected_iterations: Option<usize>,
    pub memory_estimate: Option<usize>,
    pub reasoning: String,
}

/// AI agent for mathematical problem solving
pub struct MathAgent {
    client: Client<async_openai::config::OpenAIConfig>,
    config: AgentConfig,
}

impl MathAgent {
    /// Create a new math agent with configuration
    pub fn new(config: AgentConfig) -> Result<Self> {
        let client = Client::with_config(
            async_openai::config::OpenAIConfig::new()
                .with_api_key(&config.api_key)
        );
        
        Ok(Self { client, config })
    }

    /// Analyze a mathematical problem and suggest solution strategy
    pub async fn analyze_problem(&self, problem: &MathProblem) -> Result<SolutionStrategy> {
        let system_prompt = self.create_system_prompt();
        let user_prompt = self.create_problem_prompt(problem);

        let request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessage {
                        content: system_prompt,
                        name: None,
                    }
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: user_prompt,
                        name: None,
                    }
                ),
            ],
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            ..Default::default()
        };

        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.client.chat().create(request)
        ).await
        .map_err(|_| OpenAlgebraError::IoError(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "OpenAI API timeout")
        ))??;

        self.parse_strategy_response(&response)
    }

    /// Optimize solver parameters using AI feedback
    pub async fn optimize_solver_parameters(
        &self,
        problem: &MathProblem,
        current_performance: &SolverPerformance,
    ) -> Result<HashMap<String, f64>> {
        let prompt = format!(
            "Given the following linear algebra problem and current solver performance, \
             suggest optimized parameters:\n\n\
             Problem: {}\n\
             Current Performance:\n\
             - Iterations: {}\n\
             - Residual: {:.2e}\n\
             - Time: {:.2f}s\n\
             - Memory: {} MB\n\n\
             Suggest parameter improvements in JSON format.",
            serde_json::to_string_pretty(problem)?,
            current_performance.iterations,
            current_performance.residual_norm,
            current_performance.solve_time,
            current_performance.memory_usage / 1024 / 1024
        );

        let request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: prompt,
                        name: None,
                    }
                ),
            ],
            max_tokens: self.config.max_tokens,
            temperature: Some(0.3), // Lower temperature for parameter optimization
            ..Default::default()
        };

        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.client.chat().create(request)
        ).await
        .map_err(|_| OpenAlgebraError::IoError(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "OpenAI API timeout")
        ))??;

        self.parse_parameters_response(&response)
    }

    /// Generate code for solving a specific problem
    pub async fn generate_solution_code(&self, problem: &MathProblem) -> Result<String> {
        let prompt = format!(
            "Generate Rust code using the OpenAlgebra library to solve this problem:\n\n{}\n\n\
             Include:\n\
             - Matrix creation and setup\n\
             - Appropriate solver selection\n\
             - Error handling\n\
             - Performance monitoring\n\
             - Result validation",
            serde_json::to_string_pretty(problem)?
        );

        let request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessage {
                        content: "You are an expert in numerical linear algebra and Rust programming. \
                                 Generate efficient, well-documented code using the OpenAlgebra library.".to_string(),
                        name: None,
                    }
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: prompt,
                        name: None,
                    }
                ),
            ],
            max_tokens: Some(4096),
            temperature: Some(0.2),
            ..Default::default()
        };

        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.client.chat().create(request)
        ).await
        .map_err(|_| OpenAlgebraError::IoError(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "OpenAI API timeout")
        ))??;

        Ok(response.choices[0].message.content.clone().unwrap_or_default())
    }

    /// Explain solver behavior and suggest improvements
    pub async fn explain_solver_behavior(
        &self,
        solver_name: &str,
        convergence_history: &[f64],
        problem_properties: &HashMap<String, f64>,
    ) -> Result<String> {
        let prompt = format!(
            "Analyze the convergence behavior of the {} solver:\n\n\
             Convergence History: {:?}\n\
             Problem Properties: {}\n\n\
             Explain:\n\
             1. Why the solver behaved this way\n\
             2. What the convergence pattern indicates\n\
             3. Potential improvements or alternative approaches\n\
             4. Expected performance for similar problems",
            solver_name,
            convergence_history,
            serde_json::to_string_pretty(problem_properties)?
        );

        let request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessage {
                        content: "You are an expert in numerical analysis and iterative solvers. \
                                 Provide detailed, educational explanations of solver behavior.".to_string(),
                        name: None,
                    }
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: prompt,
                        name: None,
                    }
                ),
            ],
            max_tokens: Some(2048),
            temperature: Some(0.5),
            ..Default::default()
        };

        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.client.chat().create(request)
        ).await
        .map_err(|_| OpenAlgebraError::IoError(
            std::io::Error::new(std::io::ErrorKind::TimedOut, "OpenAI API timeout")
        ))??;

        Ok(response.choices[0].message.content.clone().unwrap_or_default())
    }

    fn create_system_prompt(&self) -> String {
        "You are an expert in numerical linear algebra, sparse matrix computations, and iterative solvers. \
         Your role is to analyze mathematical problems and recommend optimal solution strategies using \
         the OpenAlgebra library. Consider factors like:\n\
         - Matrix properties (size, sparsity, condition number, symmetry)\n\
         - Solver characteristics (convergence rate, memory usage, stability)\n\
         - Preconditioner effectiveness\n\
         - Computational complexity and scalability\n\
         - Numerical accuracy requirements\n\n\
         Provide recommendations in JSON format with clear reasoning.".to_string()
    }

    fn create_problem_prompt(&self, problem: &MathProblem) -> String {
        format!(
            "Analyze this mathematical problem and recommend a solution strategy:\n\n{}\n\n\
             Provide your recommendation in the following JSON format:\n\
             {{\n\
               \"solver_type\": \"ConjugateGradient|GMRES|BiCGSTAB\",\n\
               \"preconditioner\": \"Jacobi|ILU|AMG|None\",\n\
               \"parameters\": {{\n\
                 \"tolerance\": 1e-6,\n\
                 \"max_iterations\": 1000,\n\
                 \"restart\": 30\n\
               }},\n\
               \"expected_iterations\": 50,\n\
               \"memory_estimate\": 1048576,\n\
               \"reasoning\": \"Detailed explanation of the recommendation\"\n\
             }}",
            serde_json::to_string_pretty(problem).unwrap_or_default()
        )
    }

    fn parse_strategy_response(&self, response: &CreateChatCompletionResponse) -> Result<SolutionStrategy> {
        let content = response.choices[0].message.content.clone().unwrap_or_default();
        
        // Try to extract JSON from the response
        let json_start = content.find('{').unwrap_or(0);
        let json_end = content.rfind('}').map(|i| i + 1).unwrap_or(content.len());
        let json_str = &content[json_start..json_end];
        
        match serde_json::from_str::<SolutionStrategy>(json_str) {
            Ok(strategy) => Ok(strategy),
            Err(_) => {
                // Fallback: create a basic strategy from the text
                Ok(SolutionStrategy {
                    solver_type: "ConjugateGradient".to_string(),
                    preconditioner: Some("Jacobi".to_string()),
                    parameters: HashMap::new(),
                    expected_iterations: Some(100),
                    memory_estimate: None,
                    reasoning: content,
                })
            }
        }
    }

    fn parse_parameters_response(&self, response: &CreateChatCompletionResponse) -> Result<HashMap<String, f64>> {
        let content = response.choices[0].message.content.clone().unwrap_or_default();
        
        // Try to extract JSON from the response
        let json_start = content.find('{').unwrap_or(0);
        let json_end = content.rfind('}').map(|i| i + 1).unwrap_or(content.len());
        let json_str = &content[json_start..json_end];
        
        match serde_json::from_str::<HashMap<String, f64>>(json_str) {
            Ok(params) => Ok(params),
            Err(_) => {
                // Fallback: return default parameters
                let mut params = HashMap::new();
                params.insert("tolerance".to_string(), 1e-6);
                params.insert("max_iterations".to_string(), 1000.0);
                Ok(params)
            }
        }
    }
}

/// Performance metrics for solver evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverPerformance {
    pub iterations: usize,
    pub residual_norm: f64,
    pub solve_time: f64,
    pub memory_usage: usize,
    pub convergence_rate: f64,
}

/// AI-guided solver that adapts based on problem characteristics
pub struct AdaptiveSolver {
    agent: MathAgent,
    performance_history: Vec<SolverPerformance>,
}

impl AdaptiveSolver {
    /// Create a new adaptive solver with AI guidance
    pub fn new(agent_config: AgentConfig) -> Result<Self> {
        let agent = MathAgent::new(agent_config)?;
        Ok(Self {
            agent,
            performance_history: Vec::new(),
        })
    }

    /// Solve a linear system with AI-guided strategy selection
    pub async fn solve_adaptive<T>(
        &mut self,
        matrix: &CSRMatrix<T>,
        b: &[T],
        x: &mut [T],
        problem_description: &MathProblem,
    ) -> Result<crate::solvers::SolverInfo>
    where
        T: num_traits::Float + Send + Sync + std::fmt::Debug + Clone + Default,
    {
        // Get AI recommendation
        let strategy = self.agent.analyze_problem(problem_description).await?;
        
        // Apply the recommended strategy
        let start_time = std::time::Instant::now();
        let info = match strategy.solver_type.as_str() {
            "ConjugateGradient" => {
                if let Some(precond_type) = &strategy.preconditioner {
                    match precond_type.as_str() {
                        "Jacobi" => {
                            let precond = JacobiPreconditioner::new(matrix);
                            let solver = ConjugateGradient::with_preconditioner(Box::new(precond));
                            solver.solve(matrix, b, x)
                        }
                        "ILU" => {
                            let fill_level = strategy.parameters.get("fill_level").unwrap_or(&0.0) as usize;
                            let drop_tol = strategy.parameters.get("drop_tolerance").unwrap_or(&1e-6);
                            let precond = ILUPreconditioner::new(matrix, fill_level, *drop_tol);
                            let solver = ConjugateGradient::with_preconditioner(Box::new(precond));
                            solver.solve(matrix, b, x)
                        }
                        _ => {
                            let solver = ConjugateGradient::new();
                            solver.solve(matrix, b, x)
                        }
                    }
                } else {
                    let solver = ConjugateGradient::new();
                    solver.solve(matrix, b, x)
                }
            }
            "GMRES" => {
                let restart = strategy.parameters.get("restart").unwrap_or(&30.0) as usize;
                let solver = GMRES::new(restart);
                solver.solve(matrix, b, x)
            }
            "BiCGSTAB" => {
                let solver = BiCGSTAB::new();
                solver.solve(matrix, b, x)
            }
            _ => {
                let solver = ConjugateGradient::new();
                solver.solve(matrix, b, x)
            }
        };
        
        let solve_time = start_time.elapsed().as_secs_f64();
        
        // Record performance
        let performance = SolverPerformance {
            iterations: info.iterations,
            residual_norm: info.residual_norm,
            solve_time,
            memory_usage: std::mem::size_of_val(matrix) + std::mem::size_of_val(b) + std::mem::size_of_val(x),
            convergence_rate: if info.iterations > 1 {
                (info.residual_norm / 1.0).ln() / info.iterations as f64
            } else {
                0.0
            },
        };
        
        self.performance_history.push(performance);
        
        Ok(info)
    }

    /// Get performance insights from AI
    pub async fn get_performance_insights(&self) -> Result<String> {
        if self.performance_history.is_empty() {
            return Ok("No performance data available yet.".to_string());
        }

        let latest = &self.performance_history[self.performance_history.len() - 1];
        let convergence_history: Vec<f64> = self.performance_history
            .iter()
            .map(|p| p.residual_norm)
            .collect();

        let mut properties = HashMap::new();
        properties.insert("avg_iterations".to_string(), 
            self.performance_history.iter().map(|p| p.iterations as f64).sum::<f64>() / self.performance_history.len() as f64);
        properties.insert("avg_solve_time".to_string(),
            self.performance_history.iter().map(|p| p.solve_time).sum::<f64>() / self.performance_history.len() as f64);

        self.agent.explain_solver_behavior(
            "Adaptive",
            &convergence_history,
            &properties,
        ).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::COOMatrix;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_problem_serialization() {
        let problem = MathProblem {
            problem_type: ProblemType::LinearSystem,
            description: "Test problem".to_string(),
            constraints: vec!["positive definite".to_string()],
            objectives: vec!["minimize iterations".to_string()],
            matrix_size: Some((1000, 1000)),
            sparsity_pattern: Some("tridiagonal".to_string()),
            numerical_properties: {
                let mut props = HashMap::new();
                props.insert("condition_number".to_string(), 100.0);
                props
            },
        };

        let json = serde_json::to_string(&problem).unwrap();
        let deserialized: MathProblem = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.description, "Test problem");
        assert_eq!(deserialized.matrix_size, Some((1000, 1000)));
    }

    #[tokio::test]
    async fn test_math_agent_creation() {
        let config = AgentConfig {
            api_key: "test_key".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: Some(1024),
            temperature: Some(0.5),
            timeout_seconds: 10,
        };

        let agent = MathAgent::new(config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_solver_performance() {
        let performance = SolverPerformance {
            iterations: 50,
            residual_norm: 1e-8,
            solve_time: 0.1,
            memory_usage: 1024 * 1024,
            convergence_rate: -0.2,
        };

        assert_eq!(performance.iterations, 50);
        assert!(performance.residual_norm < 1e-6);
        assert!(performance.solve_time > 0.0);
    }
} 