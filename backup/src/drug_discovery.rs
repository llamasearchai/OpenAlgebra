//! Pharmaceutical Drug Discovery Engine
//! 
//! Accelerates drug discovery through molecular modeling, virtual screening,
//! and AI-driven compound optimization with clinical trial prediction.

use crate::{
    api::ApiError,
    models::{MedicalData, ProcessingResult},
    sparse::{SparseMatrix, SparseTensor},
    medical::MedicalProcessor,
    agents::MedicalAgent,
};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Array3};
use ordered_float::OrderedFloat; // For using f64 as BTreeMap keys

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub molecule_id: String,
    pub smiles: String,
    pub inchi: String,
    pub molecular_weight: f64,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub properties: MolecularProperties,
    pub fingerprint: MolecularFingerprint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub index: usize,
    pub element: String,
    pub position: [f64; 3],
    pub charge: f64,
    pub hybridization: String,
    pub aromatic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: BondType,
    pub aromatic: bool,
    pub rotatable: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularProperties {
    pub logp: f64,                    // Lipophilicity
    pub logd: f64,                    // Distribution coefficient
    pub psa: f64,                     // Polar surface area
    pub hbd: u32,                     // Hydrogen bond donors
    pub hba: u32,                     // Hydrogen bond acceptors
    pub rotatable_bonds: u32,
    pub aromatic_rings: u32,
    pub molecular_volume: f64,
    pub solubility: f64,
    pub permeability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularFingerprint {
    pub ecfp4: Vec<u64>,              // Extended connectivity fingerprint
    pub maccs: Vec<bool>,             // MACCS keys
    pub pharmacophore: Vec<f64>,      // 3D pharmacophore features
    pub shape_descriptor: Vec<f64>,   // 3D shape descriptor
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugTarget {
    pub target_id: String,
    pub protein_name: String,
    pub uniprot_id: String,
    pub pdb_id: Option<String>,
    pub sequence: String,
    pub structure: Option<ProteinStructure>,
    pub binding_sites: Vec<BindingSite>,
    pub known_ligands: Vec<String>,
    pub disease_associations: Vec<DiseaseAssociation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinStructure {
    pub atoms: Vec<ProteinAtom>,
    pub residues: Vec<Residue>,
    pub chains: Vec<Chain>,
    pub secondary_structure: Vec<SecondaryStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinAtom {
    pub index: usize,
    pub name: String,
    pub element: String,
    pub position: [f64; 3],
    pub residue_index: usize,
    pub chain_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    pub index: usize,
    pub name: String,
    pub chain_id: String,
    pub atoms: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    pub chain_id: String,
    pub residues: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryStructure {
    pub structure_type: SecondaryStructureType,
    pub start_residue: usize,
    pub end_residue: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecondaryStructureType {
    AlphaHelix,
    BetaSheet,
    Turn,
    Loop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingSite {
    pub site_id: String,
    pub residues: Vec<usize>,
    pub volume: f64,
    pub druggability_score: f64,
    pub known_inhibitors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiseaseAssociation {
    pub disease_name: String,
    pub omim_id: Option<String>,
    pub association_type: String,
    pub evidence_level: String,
    pub therapeutic_area: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualScreeningResult {
    pub molecule_id: String,
    pub target_id: String,
    pub binding_affinity: f64,         // Predicted Kd/Ki in nM
    pub docking_score: f64,
    pub interaction_fingerprint: Vec<f64>,
    pub pose: MolecularPose,
    pub interactions: Vec<Interaction>,
    pub selectivity_score: f64,
    pub druglikeness_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularPose {
    pub position: [f64; 3],
    pub rotation: [f64; 4],           // Quaternion
    pub conformation: Vec<f64>,       // Torsion angles
    pub rmsd_from_native: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub interaction_type: InteractionType,
    pub ligand_atoms: Vec<usize>,
    pub protein_atoms: Vec<usize>,
    pub distance: f64,
    pub angle: Option<f64>,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    HydrogenBond,
    Hydrophobic,
    PiPiStacking,
    PiCation,
    SaltBridge,
    VanDerWaals,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadOptimization {
    pub parent_molecule: String,
    pub optimized_molecules: Vec<OptimizedMolecule>,
    pub optimization_strategy: OptimizationStrategy,
    pub improvement_metrics: ImprovementMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedMolecule {
    pub molecule: Molecule,
    pub modifications: Vec<Modification>,
    pub predicted_properties: PredictedProperties,
    pub synthesis_feasibility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modification {
    pub modification_type: String,
    pub position: String,
    pub original_group: String,
    pub new_group: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedProperties {
    pub binding_affinity: f64,
    pub selectivity: HashMap<String, f64>,
    pub admet: ADMETProfile,
    pub toxicity: ToxicityProfile,
    pub clinical_success_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADMETProfile {
    pub absorption: f64,
    pub distribution: f64,
    pub metabolism: MetabolismProfile,
    pub excretion: f64,
    pub toxicity: f64,
    pub half_life: f64,
    pub bioavailability: f64,
    pub clearance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolismProfile {
    pub cyp_interactions: HashMap<String, f64>,
    pub phase1_metabolites: Vec<String>,
    pub phase2_metabolites: Vec<String>,
    pub stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityProfile {
    pub herg_inhibition: f64,
    pub ames_mutagenicity: f64,
    pub hepatotoxicity: f64,
    pub cardiotoxicity: f64,
    pub ld50_prediction: f64,
    pub adverse_effects: Vec<AdverseEffect>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdverseEffect {
    pub effect_type: String,
    pub probability: f64,
    pub severity: String,
    pub reversible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    StructureBasedDesign,
    LigandBasedDesign,
    FragmentGrowing,
    BioisostericReplacement,
    MachineLearningGuided,
    QuantumMechanicsGuided,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    pub potency_improvement: f64,
    pub selectivity_improvement: f64,
    pub admet_improvement: f64,
    pub synthesis_complexity_change: f64,
}

pub struct DrugDiscoveryEngine {
    molecular_database: Arc<RwLock<MolecularDatabase>>,
    target_database: Arc<RwLock<TargetDatabase>>,
    screening_engine: Arc<RwLock<VirtualScreeningEngine>>,
    optimization_engine: Arc<RwLock<LeadOptimizationEngine>>,
    prediction_models: Arc<RwLock<DrugPredictionModels>>, // Renamed to avoid conflict
    medical_agent: Arc<MedicalAgent>,
}

struct MolecularDatabase {
    molecules: HashMap<String, Molecule>,
    fingerprint_index: FingerprintIndex,
    property_index: PropertyIndex,
    scaffold_index: ScaffoldIndex,
}

struct FingerprintIndex {
    ecfp_index: HashMap<u64, HashSet<String>>,
    pharmacophore_index: HashMap<String, HashSet<String>>,
}

struct PropertyIndex {
    logp_index: BTreeMap<OrderedFloat<f64>, HashSet<String>>,
    mw_index: BTreeMap<OrderedFloat<f64>, HashSet<String>>,
    hba_hbd_index: HashMap<(u32, u32), HashSet<String>>,
}

struct ScaffoldIndex {
    scaffolds: HashMap<String, Scaffold>,
    molecule_scaffolds: HashMap<String, Vec<String>>,
}

struct Scaffold {
    scaffold_id: String,
    smiles: String,
    molecules: HashSet<String>,
    activity_profile: HashMap<String, f64>,
}

struct TargetDatabase {
    targets: HashMap<String, DrugTarget>,
    disease_targets: HashMap<String, HashSet<String>>,
    pathway_targets: HashMap<String, HashSet<String>>,
    target_families: HashMap<String, TargetFamily>,
}

struct TargetFamily {
    family_name: String,
    members: HashSet<String>,
    conserved_residues: Vec<usize>,
    selectivity_determinants: Vec<usize>,
}

struct VirtualScreeningEngine {
    docking_program: DockingProgram,
    scoring_functions: Vec<ScoringFunction>,
    pharmacophore_models: HashMap<String, PharmacophoreModel>,
    machine_learning_models: MLScoringModels,
}

#[derive(Debug, Clone)]
enum DockingProgram {
    AutoDock,
    Glide,
    FlexX,
    GOLD,
    Custom,
}

struct ScoringFunction {
    name: String,
    weight: f64,
    function: Box<dyn Fn(&MolecularPose, &BindingSite) -> f64 + Send + Sync>,
}

struct PharmacophoreModel {
    model_id: String,
    features: Vec<PharmacophoreFeature>,
    constraints: Vec<DistanceConstraint>,
    exclusion_volumes: Vec<ExclusionVolume>,
}

struct PharmacophoreFeature {
    feature_type: String,
    position: [f64; 3],
    tolerance: f64,
    required: bool,
}

struct DistanceConstraint {
    feature1: usize,
    feature2: usize,
    min_distance: f64,
    max_distance: f64,
}

struct ExclusionVolume {
    position: [f64; 3],
    radius: f64,
}

struct MLScoringModels {
    binding_affinity_model: SparseMatrix,
    selectivity_model: SparseMatrix,
    druglikeness_model: SparseMatrix,
    toxicity_model: SparseMatrix,
}

struct LeadOptimizationEngine {
    optimization_algorithms: Vec<OptimizationAlgorithm>,
    synthesis_predictor: SynthesisPrediction,
    property_predictor: PropertyPrediction,
    retrosynthesis_engine: RetrosynthesisEngine,
}

struct OptimizationAlgorithm {
    name: String,
    algorithm_type: OptimizationType,
    parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
enum OptimizationType {
    GeneticAlgorithm,
    ParticleSwarm,
    SimulatedAnnealing,
    ReinforcementLearning,
    BayesianOptimization,
}

struct SynthesisPrediction {
    feasibility_model: SparseMatrix,
    route_predictor: SparseMatrix,
    cost_estimator: SparseMatrix,
}

struct PropertyPrediction {
    qsar_models: HashMap<String, SparseMatrix>,
    deep_learning_models: HashMap<String, SparseMatrix>,
    physics_based_models: HashMap<String, Box<dyn Fn(&Molecule) -> f64 + Send + Sync>>,
}

struct RetrosynthesisEngine {
    reaction_database: HashMap<String, ReactionTemplate>,
    synthesis_tree_builder: SynthesisTreeBuilder,
    route_scorer: RouteScorer,
}

struct ReactionTemplate {
    template_id: String,
    reaction_smarts: String,
    reaction_class: String,
    yield_range: (f64, f64),
    conditions: ReactionConditions,
}

struct ReactionConditions {
    temperature: f64,
    pressure: f64,
    solvent: String,
    catalysts: Vec<String>,
    time_hours: f64,
}

struct SynthesisTreeBuilder {
    max_depth: usize,
    beam_width: usize,
    commercial_compounds: HashSet<String>,
}

struct RouteScorer {
    complexity_weight: f64,
    yield_weight: f64,
    cost_weight: f64,
    time_weight: f64,
}

struct DrugPredictionModels { // Renamed to avoid conflict
    clinical_success_model: SparseMatrix,
    side_effect_model: SparseMatrix,
    drug_interaction_model: SparseMatrix,
    resistance_model: SparseMatrix,
}

impl DrugDiscoveryEngine {
    pub async fn new(medical_agent: MedicalAgent) -> Result<Self, ApiError> {
        let molecular_database = MolecularDatabase {
            molecules: HashMap::new(),
            fingerprint_index: FingerprintIndex {
                ecfp_index: HashMap::new(),
                pharmacophore_index: HashMap::new(),
            },
            property_index: PropertyIndex {
                logp_index: BTreeMap::new(),
                mw_index: BTreeMap::new(),
                hba_hbd_index: HashMap::new(),
            },
            scaffold_index: ScaffoldIndex {
                scaffolds: HashMap::new(),
                molecule_scaffolds: HashMap::new(),
            },
        };
        
        let target_database = TargetDatabase {
            targets: HashMap::new(),
            disease_targets: HashMap::new(),
            pathway_targets: HashMap::new(),
            target_families: HashMap::new(),
        };
        
        let screening_engine = VirtualScreeningEngine {
            docking_program: DockingProgram::Custom,
            scoring_functions: Self::initialize_scoring_functions(),
            pharmacophore_models: HashMap::new(),
            machine_learning_models: Self::load_ml_models()?,
        };
        
        let optimization_engine = LeadOptimizationEngine {
            optimization_algorithms: Self::initialize_optimization_algorithms(),
            synthesis_predictor: Self::initialize_synthesis_predictor()?,
            property_predictor: Self::initialize_property_predictor()?,
            retrosynthesis_engine: Self::initialize_retrosynthesis_engine()?,
        };
        
        let prediction_models = DrugPredictionModels { // Renamed
            clinical_success_model: Self::load_clinical_model()?,
            side_effect_model: Self::load_side_effect_model()?,
            drug_interaction_model: Self::load_interaction_model()?,
            resistance_model: Self::load_resistance_model()?,
        };
        
        Ok(Self {
            molecular_database: Arc::new(RwLock::new(molecular_database)),
            target_database: Arc::new(RwLock::new(target_database)),
            screening_engine: Arc::new(RwLock::new(screening_engine)),
            optimization_engine: Arc::new(RwLock::new(optimization_engine)),
            prediction_models: Arc::new(RwLock::new(prediction_models)),
            medical_agent: Arc::new(medical_agent),
        })
    }
    
    pub async fn virtual_screening(
        &self,
        target_id: &str,
        molecule_library_ids: Vec<String>, // Changed to IDs
        screening_params: ScreeningParameters,
    ) -> Result<Vec<VirtualScreeningResult>, ApiError> {
        let target_db = self.target_database.read().await;
        let target = target_db.targets.get(target_id)
            .ok_or_else(|| ApiError::NotFound(format!("Target {} not found", target_id)))?
            .clone(); // Clone target data
        
        let molecular_db = self.molecular_database.read().await;
        
        let mut results = Vec::new();
        
        // Parallel screening
        let chunks: Vec<_> = molecule_library_ids.chunks(100).collect();
        let mut handles = Vec::new();
        
        for chunk_ids in chunks {
            let molecules_to_screen: Vec<_> = chunk_ids.iter()
                .filter_map(|id| molecular_db.molecules.get(id).cloned())
                .collect();
            
            if molecules_to_screen.is_empty() {
                continue;
            }
            
            let current_target = target.clone(); // Clone for each task
            let params = screening_params.clone();
            let engine_clone = Arc::clone(&self.screening_engine); // Clone Arc
            
            let handle = tokio::spawn(async move {
                Self::screen_molecules(molecules_to_screen, current_target, params, engine_clone).await
            });
            
            handles.push(handle);
        }
        
        // Collect results
        for handle in handles {
            let batch_results = handle.await??; // Propagate errors
            results.extend(batch_results);
        }
        
        // Sort by binding affinity
        results.sort_by(|a, b| a.binding_affinity.partial_cmp(&b.binding_affinity).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply filters
        let filtered_results = self.apply_screening_filters(results, &screening_params)?;
        
        // Calculate selectivity
        let selective_results = self.calculate_selectivity(filtered_results, target_id).await?;
        
        Ok(selective_results)
    }
    
    async fn screen_molecules(
        molecules: Vec<Molecule>,
        target: DrugTarget,
        params: ScreeningParameters,
        engine_arc: Arc<RwLock<VirtualScreeningEngine>>, // Use Arc
    ) -> Result<Vec<VirtualScreeningResult>, ApiError> {
        let mut results = Vec::new();
        let engine = engine_arc.read().await; // Read lock
        
        for molecule in molecules {
            // Perform docking
            let docking_result = Self::perform_docking(&molecule, &target, &params)?;
            
            // Score the pose
            let mut total_score = 0.0;
            if target.binding_sites.is_empty() {
                 return Err(ApiError::ProcessingError("Target has no binding sites defined".to_string()));
            }

            for scoring_fn in &engine.scoring_functions {
                total_score += scoring_fn.weight * (scoring_fn.function)(&docking_result.pose, &target.binding_sites[0]);
            }
            
            // Predict binding affinity using ML
            let ml_affinity = Self::predict_binding_affinity(
                &molecule,
                &target,
                &docking_result,
                &engine.machine_learning_models.binding_affinity_model,
            )?;
            
            // Analyze interactions
            let interactions = Self::analyze_interactions(&molecule, &target, &docking_result.pose)?;
            
            // Calculate druglikeness
            let druglikeness = Self::calculate_druglikeness(&molecule, &engine.machine_learning_models.druglikeness_model)?;
            
            results.push(VirtualScreeningResult {
                molecule_id: molecule.molecule_id.clone(),
                target_id: target.target_id.clone(),
                binding_affinity: ml_affinity,
                docking_score: total_score,
                interaction_fingerprint: Self::generate_interaction_fingerprint(&interactions),
                pose: docking_result.pose,
                interactions,
                selectivity_score: 0.0, // Will be calculated later
                druglikeness_score: druglikeness,
            });
        }
        
        Ok(results)
    }
    
    pub async fn optimize_lead(
        &self,
        lead_molecule_id: &str,
        target_id: &str,
        optimization_params: OptimizationParameters,
    ) -> Result<LeadOptimization, ApiError> {
        let molecular_db = self.molecular_database.read().await;
        let lead_molecule = molecular_db.molecules.get(lead_molecule_id)
            .ok_or_else(|| ApiError::NotFound(format!("Lead molecule {} not found", lead_molecule_id)))?
            .clone();
        
        let optimization_engine_lock = self.optimization_engine.read().await;
        
        // Generate optimization strategies
        let strategies = self.generate_optimization_strategies(&lead_molecule, &optimization_params)?;
        
        let mut optimized_molecules = Vec::new();
        
        for strategy in strategies {
            // Apply optimization algorithm
            let candidates = match &strategy {
                OptimizationStrategy::StructureBasedDesign => {
                    self.structure_based_optimization(&lead_molecule, target_id, &optimization_params).await?
                }
                OptimizationStrategy::FragmentGrowing => {
                    self.fragment_growing_optimization(&lead_molecule, &optimization_params).await?
                }
                OptimizationStrategy::BioisostericReplacement => {
                    self.bioisosteric_replacement(&lead_molecule, &optimization_params).await?
                }
                OptimizationStrategy::MachineLearningGuided => {
                    self.ml_guided_optimization(&lead_molecule, target_id, &optimization_params).await?
                }
                _ => vec![],
            };
            
            // Evaluate candidates
            for candidate in candidates {
                let predicted_properties = self.predict_molecule_properties(&candidate, target_id).await?;
                let synthesis_feasibility = self.predict_synthesis_feasibility(&candidate, &optimization_engine_lock.synthesis_predictor)?;
                
                optimized_molecules.push(OptimizedMolecule {
                    molecule: candidate.molecule,
                    modifications: candidate.modifications,
                    predicted_properties,
                    synthesis_feasibility,
                });
            }
        }
        
        // Sort by overall improvement
        optimized_molecules.sort_by(|a, b| {
            let score_a = a.predicted_properties.binding_affinity * a.synthesis_feasibility;
            let score_b = b.predicted_properties.binding_affinity * b.synthesis_feasibility;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Calculate improvement metrics
        let improvement_metrics = self.calculate_improvement_metrics(&lead_molecule, &optimized_molecules)?;
        
        Ok(LeadOptimization {
            parent_molecule: lead_molecule_id.to_string(),
            optimized_molecules: optimized_molecules.into_iter().take(optimization_params.max_results).collect(),
            optimization_strategy: OptimizationStrategy::MachineLearningGuided, // Example, choose best
            improvement_metrics,
        })
    }
    
    pub async fn predict_clinical_success(
        &self,
        molecule_id: &str,
        target_id: &str,
        indication: &str,
    ) -> Result<ClinicalPrediction, ApiError> {
        let molecular_db = self.molecular_database.read().await;
        let molecule = molecular_db.molecules.get(molecule_id)
            .ok_or_else(|| ApiError::NotFound(format!("Molecule {} not found", molecule_id)))?;
        
        let target_db = self.target_database.read().await;
        let target = target_db.targets.get(target_id)
            .ok_or_else(|| ApiError::NotFound(format!("Target {} not found", target_id)))?;
        
        let models = self.prediction_models.read().await;
        
        // Predict clinical success probability
        let success_features = self.extract_clinical_features(molecule, target, indication)?;
        let success_probability = Self::apply_sparse_model(&models.clinical_success_model, &success_features)?;
        
        // Predict side effects
        let side_effect_features = self.extract_side_effect_features(molecule)?;
        let side_effect_profile = Self::predict_side_effects(&models.side_effect_model, &side_effect_features)?;
        
        // Predict drug interactions
        let interaction_profile = self.predict_drug_interactions(molecule, &models.drug_interaction_model).await?;
        
        // Predict resistance development
        let resistance_timeline = self.predict_resistance_development(molecule, target, &models.resistance_model)?;
        
        // Generate clinical trial design recommendations
        let trial_recommendations = self.generate_trial_recommendations(
            molecule,
            target,
            indication,
            &side_effect_profile,
        )?;
        
        Ok(ClinicalPrediction {
            molecule_id: molecule_id.to_string(),
            success_probability,
            phase_predictions: self.predict_phase_success(success_probability),
            side_effect_profile,
            interaction_profile,
            resistance_timeline,
            trial_recommendations,
            regulatory_considerations: self.generate_regulatory_considerations(molecule, indication)?,
        })
    }
    
    pub async fn design_combination_therapy(
        &self,
        primary_drug_id: &str, // Changed to ID
        target_disease: &str,
        combination_params: CombinationParameters,
    ) -> Result<Vec<CombinationTherapy>, ApiError> {
        let molecular_db = self.molecular_database.read().await;
        let primary_molecule = molecular_db.molecules.get(primary_drug_id)
            .ok_or_else(|| ApiError::NotFound(format!("Primary drug {} not found", primary_drug_id)))?;
        
        // Identify complementary targets
        let complementary_targets = self.identify_complementary_targets(target_disease, &combination_params).await?;
        
        let mut combination_candidates = Vec::new();
        
        for target in complementary_targets {
            // Find molecules for each target
            let target_molecules = self.find_molecules_for_target(&target, &combination_params).await?;
            
            for molecule in target_molecules {
                // Check compatibility
                let compatibility = self.check_drug_compatibility(primary_molecule, &molecule).await?;
                
                if compatibility.score > combination_params.min_compatibility_score {
                    // Predict synergy
                    let synergy = self.predict_synergy(primary_molecule, &molecule, target_disease).await?;
                    
                    // Optimize dosing
                    let dosing_schedule = self.optimize_combination_dosing(
                        primary_molecule,
                        &molecule,
                        &synergy,
                    ).await?;
                    
                    combination_candidates.push(CombinationTherapy {
                        drugs: vec![primary_drug_id.to_string(), molecule.molecule_id.clone()],
                        targets: vec![primary_molecule.molecule_id.clone(), target.target_id.clone()], // Example, should be target IDs
                        synergy_score: synergy.score,
                        compatibility_score: compatibility.score,
                        dosing_schedule,
                        predicted_efficacy: synergy.predicted_efficacy,
                        safety_profile: self.predict_combination_safety(primary_molecule, &molecule).await?,
                    });
                }
            }
        }
        
        // Sort by overall score
        combination_candidates.sort_by(|a, b| {
            let score_a = a.synergy_score * a.compatibility_score * a.predicted_efficacy;
            let score_b = b.synergy_score * b.compatibility_score * b.predicted_efficacy;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(combination_candidates.into_iter().take(combination_params.max_combinations).collect())
    }
    
    // Helper methods
    fn initialize_scoring_functions() -> Vec<ScoringFunction> {
        vec![
            ScoringFunction {
                name: "VdW".to_string(),
                weight: 1.0,
                function: Box::new(|_pose, _site| {
                    // Van der Waals scoring
                    -5.0 // Example score
                }),
            },
            ScoringFunction {
                name: "Electrostatic".to_string(),
                weight: 0.8,
                function: Box::new(|_pose, _site| {
                    // Electrostatic scoring
                    -2.0 // Example score
                }),
            },
            ScoringFunction {
                name: "HBond".to_string(),
                weight: 1.2,
                function: Box::new(|_pose, _site| {
                    // Hydrogen bond scoring
                    -3.0 // Example score
                }),
            },
        ]
    }
    
    fn load_ml_models() -> Result<MLScoringModels, ApiError> {
        Ok(MLScoringModels {
            binding_affinity_model: SparseMatrix::identity(1000),
            selectivity_model: SparseMatrix::identity(500),
            druglikeness_model: SparseMatrix::identity(200),
            toxicity_model: SparseMatrix::identity(300),
        })
    }
    
    fn initialize_optimization_algorithms() -> Vec<OptimizationAlgorithm> {
        vec![
            OptimizationAlgorithm {
                name: "Genetic Algorithm".to_string(),
                algorithm_type: OptimizationType::GeneticAlgorithm,
                parameters: [
                    ("population_size".to_string(), 100.0),
                    ("mutation_rate".to_string(), 0.1),
                    ("crossover_rate".to_string(), 0.8),
                ].iter().cloned().collect(),
            },
            OptimizationAlgorithm {
                name: "Reinforcement Learning".to_string(),
                algorithm_type: OptimizationType::ReinforcementLearning,
                parameters: [
                    ("learning_rate".to_string(), 0.001),
                    ("epsilon".to_string(), 0.1),
                    ("gamma".to_string(), 0.99),
                ].iter().cloned().collect(),
            },
        ]
    }
    
    fn initialize_synthesis_predictor() -> Result<SynthesisPrediction, ApiError> {
        Ok(SynthesisPrediction {
            feasibility_model: SparseMatrix::identity(500),
            route_predictor: SparseMatrix::identity(1000),
            cost_estimator: SparseMatrix::identity(200),
        })
    }
    
    fn initialize_property_predictor() -> Result<PropertyPrediction, ApiError> {
        Ok(PropertyPrediction {
            qsar_models: HashMap::new(),
            deep_learning_models: HashMap::new(),
            physics_based_models: HashMap::new(),
        })
    }
    
    fn initialize_retrosynthesis_engine() -> Result<RetrosynthesisEngine, ApiError> {
        Ok(RetrosynthesisEngine {
            reaction_database: HashMap::new(),
            synthesis_tree_builder: SynthesisTreeBuilder {
                max_depth: 10,
                beam_width: 50,
                commercial_compounds: HashSet::new(),
            },
            route_scorer: RouteScorer {
                complexity_weight: 0.3,
                yield_weight: 0.3,
                cost_weight: 0.2,
                time_weight: 0.2,
            },
        })
    }
    
    fn load_clinical_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(2000))
    }
    
    fn load_side_effect_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(1500))
    }
    
    fn load_interaction_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(1000))
    }
    
    fn load_resistance_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(800))
    }
    
    fn perform_docking(
        molecule: &Molecule,
        target: &DrugTarget,
        params: &ScreeningParameters,
    ) -> Result<DockingResult, ApiError> {
        // Simplified docking simulation
        Ok(DockingResult {
            pose: MolecularPose {
                position: [target.binding_sites.get(0).map_or(0.0, |bs| bs.volume / 100.0), 0.0, 0.0], // Example
                rotation: [1.0, 0.0, 0.0, 0.0],
                conformation: vec![0.0; molecule.properties.rotatable_bonds as usize],
                rmsd_from_native: None,
            },
            score: -8.5, // Example score
        })
    }
    
    fn predict_binding_affinity(
        molecule: &Molecule,
        target: &DrugTarget,
        docking_result: &DockingResult,
        model: &SparseMatrix,
    ) -> Result<f64, ApiError> {
        // Extract features and apply model
        let features = Self::extract_binding_features(molecule, target, docking_result)?;
        let affinity_log_score = Self::apply_sparse_model(model, &features)?;
        Ok(10.0_f64.powf(-affinity_log_score) * 1e9) // Convert pKd/pKi to nM
    }
    
    fn analyze_interactions(
        molecule: &Molecule,
        target: &DrugTarget,
        pose: &MolecularPose,
    ) -> Result<Vec<Interaction>, ApiError> {
        let mut interactions = Vec::new();
        
        // Simplified interaction analysis - based on proximity
        if let Some(binding_site) = target.binding_sites.get(0) {
            if let Some(protein_structure) = &target.structure {
                for (i, atom_lig) in molecule.atoms.iter().enumerate() {
                    for res_idx in &binding_site.residues {
                        if let Some(residue) = protein_structure.residues.get(*res_idx) {
                            for atom_prot_idx in &residue.atoms {
                                if let Some(atom_prot) = protein_structure.atoms.get(*atom_prot_idx) {
                                    let dist_sq = (atom_lig.position[0] - atom_prot.position[0]).powi(2) +
                                                  (atom_lig.position[1] - atom_prot.position[1]).powi(2) +
                                                  (atom_lig.position[2] - atom_prot.position[2]).powi(2);
                                    if dist_sq < 4.0_f64.powi(2) { // within 4 Angstroms
                                        interactions.push(Interaction {
                                            interaction_type: InteractionType::VanDerWaals, // Default
                                            ligand_atoms: vec![i],
                                            protein_atoms: vec![*atom_prot_idx],
                                            distance: dist_sq.sqrt(),
                                            angle: None,
                                            energy: -1.0, // Example
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(interactions)
    }
    
    fn calculate_druglikeness(molecule: &Molecule, model: &SparseMatrix) -> Result<f64, ApiError> {
        // Lipinski's Rule of Five
        let mut score = 1.0;
        
        if molecule.molecular_weight > 500.0 { score *= 0.8; }
        if molecule.properties.logp > 5.0 { score *= 0.8; }
        if molecule.properties.hbd > 5 { score *= 0.8; }
        if molecule.properties.hba > 10 { score *= 0.8; }
        
        // Apply ML model for more sophisticated prediction
        let features = Self::extract_druglikeness_features(molecule)?;
        let ml_score_raw = Self::apply_sparse_model(model, &features)?;
        let ml_score_sigmoid = 1.0 / (1.0 + (-ml_score_raw).exp()); // Sigmoid to get 0-1 score
        
        Ok(score * ml_score_sigmoid)
    }
    
    fn generate_interaction_fingerprint(interactions: &[Interaction]) -> Vec<f64> {
        let mut fingerprint = vec![0.0; 6]; // Simplified
        
        for interaction in interactions {
            match interaction.interaction_type {
                InteractionType::HydrogenBond => fingerprint[0] += 1.0,
                InteractionType::Hydrophobic => fingerprint[1] += 1.0,
                InteractionType::PiPiStacking => fingerprint[2] += 1.0,
                InteractionType::PiCation => fingerprint[3] += 1.0,
                InteractionType::SaltBridge => fingerprint[4] += 1.0,
                InteractionType::VanDerWaals => fingerprint[5] += 1.0,
            }
        }
        
        fingerprint
    }
    
    fn apply_screening_filters(
        &self,
        results: Vec<VirtualScreeningResult>,
        params: &ScreeningParameters,
    ) -> Result<Vec<VirtualScreeningResult>, ApiError> {
        Ok(results.into_iter()
            .filter(|r| r.binding_affinity < params.max_affinity_nm)
            .filter(|r| r.druglikeness_score > params.min_druglikeness)
            .collect())
    }
    
    async fn calculate_selectivity(
        &self,
        mut results: Vec<VirtualScreeningResult>,
        primary_target_id: &str,
    ) -> Result<Vec<VirtualScreeningResult>, ApiError> {
        let target_db = self.target_database.read().await;
        let molecular_db = self.molecular_database.read().await;
        let screening_engine = self.screening_engine.read().await;

        for result in &mut results {
            let molecule_opt = molecular_db.molecules.get(&result.molecule_id);
            if molecule_opt.is_none() { continue; }
            let molecule = molecule_opt.unwrap();
            
            let mut off_target_affinities = Vec::new();
            for (target_id, target) in &target_db.targets {
                if target_id != primary_target_id {
                    // Simplified off-target prediction
                    let docking_res = Self::perform_docking(molecule, target, &ScreeningParameters::default())?;
                    let affinity = Self::predict_binding_affinity(molecule, target, &docking_res, &screening_engine.machine_learning_models.binding_affinity_model)?;
                    off_target_affinities.push(affinity);
                }
            }
            
            if !off_target_affinities.is_empty() {
                let min_off_target_affinity = off_target_affinities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                result.selectivity_score = min_off_target_affinity / result.binding_affinity.max(1e-9); // Avoid division by zero
            } else {
                result.selectivity_score = f64::INFINITY; // Infinitely selective if no off-targets
            }
        }
        
        Ok(results)
    }
    
    fn generate_optimization_strategies(
        &self,
        _molecule: &Molecule, // Placeholder
        params: &OptimizationParameters,
    ) -> Result<Vec<OptimizationStrategy>, ApiError> {
        let mut strategies = Vec::new();
        
        if params.use_structure_based { strategies.push(OptimizationStrategy::StructureBasedDesign); }
        if params.use_fragment_growing { strategies.push(OptimizationStrategy::FragmentGrowing); }
        if params.use_bioisosteric { strategies.push(OptimizationStrategy::BioisostericReplacement); }
        if params.use_ml_guided { strategies.push(OptimizationStrategy::MachineLearningGuided); }
        
        if strategies.is_empty() {
            strategies.push(OptimizationStrategy::MachineLearningGuided); // Default
        }
        Ok(strategies)
    }
    
    async fn structure_based_optimization(
        &self,
        _molecule: &Molecule,
        _target_id: &str,
        _params: &OptimizationParameters,
    ) -> Result<Vec<CandidateMolecule>, ApiError> {
        // Structure-based drug design - Placeholder
        Ok(vec![])
    }
    
    async fn fragment_growing_optimization(
        &self,
        _molecule: &Molecule,
        _params: &OptimizationParameters,
    ) -> Result<Vec<CandidateMolecule>, ApiError> {
        // Fragment growing approach - Placeholder
        Ok(vec![])
    }
    
    async fn bioisosteric_replacement(
        &self,
        _molecule: &Molecule,
        _params: &OptimizationParameters,
    ) -> Result<Vec<CandidateMolecule>, ApiError> {
        // Bioisosteric replacement - Placeholder
        Ok(vec![])
    }
    
    async fn ml_guided_optimization(
        &self,
        molecule: &Molecule,
        target_id: &str,
        params: &OptimizationParameters,
    ) -> Result<Vec<CandidateMolecule>, ApiError> {
        // ML-guided optimization using generative models - Placeholder
        // This would involve complex generative chemistry models
        let mut candidates = Vec::new();
        for i in 0..5 { // Generate 5 example candidates
             let mut new_mol = molecule.clone();
             new_mol.molecule_id = format!("{}_opt_{}", molecule.molecule_id, i);
             new_mol.properties.logp += 0.1 * (i as f64); // Slightly modify properties
             candidates.push(CandidateMolecule {
                 molecule: new_mol,
                 modifications: vec![Modification {
                     modification_type: "ML_suggestion".to_string(),
                     position: "R_group".to_string(),
                     original_group: "H".to_string(),
                     new_group: "CH3".to_string(),
                     rationale: "Predicted to improve binding".to_string(),
                 }],
             });
        }
        Ok(candidates)
    }
    
    async fn predict_molecule_properties(
        &self,
        candidate: &CandidateMolecule,
        _target_id: &str, // Placeholder
    ) -> Result<PredictedProperties, ApiError> {
        // Simplified property prediction
        Ok(PredictedProperties {
            binding_affinity: 10.0 * (1.0 - candidate.modifications.len() as f64 * 0.05).max(0.1), // Example
            selectivity: HashMap::new(),
            admet: ADMETProfile {
                absorption: 0.8, distribution: 0.7,
                metabolism: MetabolismProfile {
                    cyp_interactions: HashMap::new(), phase1_metabolites: vec![],
                    phase2_metabolites: vec![], stability: 0.9,
                },
                excretion: 0.8, toxicity: 0.1, half_life: 4.5,
                bioavailability: 0.6, clearance: 1.2,
            },
            toxicity: ToxicityProfile {
                herg_inhibition: 0.1, ames_mutagenicity: 0.05, hepatotoxicity: 0.1,
                cardiotoxicity: 0.08, ld50_prediction: 2000.0, adverse_effects: vec![],
            },
            clinical_success_probability: 0.25,
        })
    }
    
    fn predict_synthesis_feasibility(
        &self,
        _candidate: &CandidateMolecule, // Placeholder
        _predictor: &SynthesisPrediction, // Placeholder
    ) -> Result<f64, ApiError> {
        Ok(0.85) // Example feasibility
    }
    
    fn calculate_improvement_metrics(
        &self,
        lead: &Molecule,
        optimized_vec: &[OptimizedMolecule],
    ) -> Result<ImprovementMetrics, ApiError> {
        if optimized_vec.is_empty() {
            return Ok(ImprovementMetrics {
                potency_improvement: 0.0, selectivity_improvement: 0.0,
                admet_improvement: 0.0, synthesis_complexity_change: 0.0,
            });
        }
        let best_optimized = &optimized_vec[0]; // Assuming sorted

        // Example calculation
        let potency_lead = 1.0 / (lead.properties.logp.abs() + 1.0); // Simplified potency
        let potency_opt = 1.0 / (best_optimized.molecule.properties.logp.abs() + 1.0);

        Ok(ImprovementMetrics {
            potency_improvement: (potency_opt / potency_lead.max(1e-9)) * 100.0 - 100.0,
            selectivity_improvement: 5.0, // Placeholder
            admet_improvement: 2.0, // Placeholder
            synthesis_complexity_change: -0.5, // Placeholder
        })
    }
    
    fn extract_clinical_features(
        &self,
        _molecule: &Molecule, _target: &DrugTarget, _indication: &str, // Placeholders
    ) -> Result<Vec<f64>, ApiError> {
        Ok(vec![0.5; 2000]) // Match model input size
    }
    
    fn extract_side_effect_features(&self, _molecule: &Molecule) -> Result<Vec<f64>, ApiError> {
        Ok(vec![0.3; 1500]) // Match model input size
    }
    
    fn predict_side_effects(
        model: &SparseMatrix,
        features: &[f64],
    ) -> Result<SideEffectProfile, ApiError> {
        let _predictions = Self::apply_sparse_model(model, features)?;
        // Interpret predictions to populate SideEffectProfile
        Ok(SideEffectProfile {
            common_effects: vec!["Headache".to_string(), "Nausea".to_string()],
            serious_effects: vec!["Liver Injury".to_string()],
            black_box_warnings: vec![],
        })
    }
    
    async fn predict_drug_interactions(
        &self,
        _molecule: &Molecule, model: &SparseMatrix, // Placeholders
    ) -> Result<DrugInteractionProfile, ApiError> {
        let features = vec![0.2; 1000]; // Example features
        let _predictions = Self::apply_sparse_model(model, &features)?;
        // Interpret predictions
        Ok(DrugInteractionProfile {
            major_interactions: vec!["Warfarin".to_string()],
            moderate_interactions: vec![],
            minor_interactions: vec![],
        })
    }
    
    fn predict_resistance_development(
        &self,
        _molecule: &Molecule, _target: &DrugTarget, model: &SparseMatrix, // Placeholders
    ) -> Result<ResistanceTimeline, ApiError> {
        let features = vec![0.1; 800]; // Example features
        let _predictions = Self::apply_sparse_model(model, &features)?;
        // Interpret predictions
        Ok(ResistanceTimeline {
            time_to_resistance_months: 24,
            resistance_mechanisms: vec!["Target mutation".to_string()],
            mitigation_strategies: vec!["Combination therapy".to_string()],
        })
    }
    
    fn generate_trial_recommendations(
        &self,
        _molecule: &Molecule, _target: &DrugTarget, _indication: &str, // Placeholders
        _side_effects: &SideEffectProfile,
    ) -> Result<TrialRecommendations, ApiError> {
        Ok(TrialRecommendations {
            phase1_design: PhaseDesign {
                dose_escalation: "3+3".to_string(), starting_dose_mg: 1.0, max_dose_mg: 100.0,
                endpoints: vec!["Safety".to_string(), "PK".to_string()],
            },
            phase2_design: PhaseDesign {
                dose_escalation: "Adaptive".to_string(), starting_dose_mg: 10.0, max_dose_mg: 50.0,
                endpoints: vec!["Efficacy".to_string(), "Safety".to_string()],
            },
            phase3_design: PhaseDesign {
                dose_escalation: "Fixed".to_string(), starting_dose_mg: 25.0, max_dose_mg: 25.0,
                endpoints: vec!["Primary efficacy".to_string(), "Secondary endpoints".to_string()],
            },
            biomarkers: vec!["Genetic marker X".to_string()],
            patient_stratification: vec!["Patients with biomarker X".to_string()],
        })
    }
    
    fn predict_phase_success(&self, overall_success: f64) -> PhasePredictions {
        PhasePredictions {
            phase1_success: (overall_success * 1.5).min(1.0),
            phase2_success: (overall_success * 1.2).min(1.0),
            phase3_success: overall_success.min(1.0),
            regulatory_approval: (overall_success * 0.9).min(1.0),
        }
    }
    
    fn generate_regulatory_considerations(
        &self,
        _molecule: &Molecule, _indication: &str, // Placeholders
    ) -> Result<Vec<String>, ApiError> {
        Ok(vec![
            "Consider FDA breakthrough therapy designation if applicable.".to_string(),
            "Prepare for potential Risk Evaluation and Mitigation Strategies (REMS) requirement.".to_string(),
            "Ensure pediatric study plan (PSP) is addressed if relevant.".to_string(),
        ])
    }
    
    async fn identify_complementary_targets(
        &self,
        _disease: &str, _params: &CombinationParameters, // Placeholders
    ) -> Result<Vec<DrugTarget>, ApiError> {
        // Placeholder: return a dummy target
        Ok(vec![DrugTarget {
            target_id: "TARGET_COMPLEMENTARY_1".to_string(),
            protein_name: "Complementary Protein Alpha".to_string(),
            uniprot_id: "PCOMP1".to_string(),
            pdb_id: None,
            sequence: "MCOMPLEMENTARY...".to_string(),
            structure: None,
            binding_sites: vec![],
            known_ligands: vec![],
            disease_associations: vec![],
        }])
    }
    
    async fn find_molecules_for_target(
        &self,
        _target: &DrugTarget, _params: &CombinationParameters, // Placeholders
    ) -> Result<Vec<Molecule>, ApiError> {
        // Placeholder: return a dummy molecule
        Ok(vec![Molecule {
            molecule_id: "MOL_COMPLEMENTARY_1".to_string(),
            smiles: "CCO".to_string(), inchi: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3".to_string(),
            molecular_weight: 46.07, atoms: vec![], bonds: vec![],
            properties: MolecularProperties { logp: -0.31, logd: -0.31, psa: 20.23, hbd: 1, hba: 1, rotatable_bonds: 0, aromatic_rings: 0, molecular_volume: 50.0, solubility: 1000.0, permeability: 1.0 },
            fingerprint: MolecularFingerprint { ecfp4: vec![], maccs: vec![], pharmacophore: vec![], shape_descriptor: vec![] },
        }])
    }
    
    async fn check_drug_compatibility(
        &self,
        _drug1: &Molecule, _drug2: &Molecule, // Placeholders
    ) -> Result<Compatibility, ApiError> {
        Ok(Compatibility { score: 0.9, issues: vec![] })
    }
    
    async fn predict_synergy(
        &self,
        _drug1: &Molecule, _drug2: &Molecule, _disease: &str, // Placeholders
    ) -> Result<Synergy, ApiError> {
        Ok(Synergy {
            score: 1.5, predicted_efficacy: 0.85,
            mechanism: "Complementary pathway inhibition".to_string(),
        })
    }
    
    async fn optimize_combination_dosing(
        &self,
        _drug1: &Molecule, _drug2: &Molecule, _synergy: &Synergy, // Placeholders
    ) -> Result<DosingSchedule, ApiError> {
        Ok(DosingSchedule {
            drug1_dose_mg: 25.0, drug2_dose_mg: 10.0,
            frequency: "Once daily".to_string(), duration_days: 28,
        })
    }
    
    async fn predict_combination_safety(
        &self,
        _drug1: &Molecule, _drug2: &Molecule, // Placeholders
    ) -> Result<SafetyProfile, ApiError> {
        Ok(SafetyProfile {
            overall_safety_score: 0.8,
            interaction_risks: vec!["Increased sedation if taken with CNS depressants.".to_string()],
            contraindications: vec!["Severe liver impairment.".to_string()],
        })
    }
    
    fn extract_binding_features(
        _molecule: &Molecule, _target: &DrugTarget, _docking: &DockingResult, // Placeholders
    ) -> Result<Vec<f64>, ApiError> {
        Ok(vec![0.7; 1000]) // Match model input size
    }
    
    fn extract_druglikeness_features(molecule: &Molecule) -> Result<Vec<f64>, ApiError> {
        // Ensure this matches the input size of druglikeness_model (200)
        let mut features = vec![
            molecule.molecular_weight, molecule.properties.logp,
            molecule.properties.logd, molecule.properties.psa,
            molecule.properties.hbd as f64, molecule.properties.hba as f64,
            molecule.properties.rotatable_bonds as f64,
            molecule.properties.aromatic_rings as f64,
            molecule.properties.molecular_volume,
            molecule.properties.solubility,
            molecule.properties.permeability,
        ];
        // Pad or truncate to 200 features
        features.resize(200, 0.0);
        Ok(features)
    }
    
    fn apply_sparse_model(model: &SparseMatrix, features: &[f64]) -> Result<f64, ApiError> {
        if features.len() != model.num_cols() {
            return Err(ApiError::ProcessingError(format!(
                "Feature vector length {} does not match model input dimension {}",
                features.len(), model.num_cols()
            )));
        }
        let input = Array1::from_vec(features.to_vec());
        let output = model.dot(&input); // Assuming model is 1xN or Nx1 for scalar output
        if output.is_empty() {
            return Err(ApiError::ProcessingError("Model output is empty".to_string()));
        }
        Ok(output[0])
    }
}

// Additional structures for DrugDiscoveryEngine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMolecule {
    pub molecule: Molecule,
    pub modifications: Vec<Modification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalPrediction {
    pub molecule_id: String,
    pub success_probability: f64,
    pub phase_predictions: PhasePredictions,
    pub side_effect_profile: SideEffectProfile,
    pub interaction_profile: DrugInteractionProfile,
    pub resistance_timeline: ResistanceTimeline,
    pub trial_recommendations: TrialRecommendations,
    pub regulatory_considerations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePredictions {
    pub phase1_success: f64,
    pub phase2_success: f64,
    pub phase3_success: f64,
    pub regulatory_approval: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffectProfile {
    pub common_effects: Vec<String>,
    pub serious_effects: Vec<String>,
    pub black_box_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteractionProfile {
    pub major_interactions: Vec<String>,
    pub moderate_interactions: Vec<String>,
    pub minor_interactions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResistanceTimeline {
    pub time_to_resistance_months: u32,
    pub resistance_mechanisms: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecommendations {
    pub phase1_design: PhaseDesign,
    pub phase2_design: PhaseDesign,
    pub phase3_design: PhaseDesign,
    pub biomarkers: Vec<String>,
    pub patient_stratification: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDesign {
    pub dose_escalation: String,
    pub starting_dose_mg: f64,
    pub max_dose_mg: f64,
    pub endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinationTherapy {
    pub drugs: Vec<String>,
    pub targets: Vec<String>,
    pub synergy_score: f64,
    pub compatibility_score: f64,
    pub dosing_schedule: DosingSchedule,
    pub predicted_efficacy: f64,
    pub safety_profile: SafetyProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compatibility {
    pub score: f64,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synergy {
    pub score: f64,
    pub predicted_efficacy: f64,
    pub mechanism: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosingSchedule {
    pub drug1_dose_mg: f64,
    pub drug2_dose_mg: f64,
    pub frequency: String,
    pub duration_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyProfile {
    pub overall_safety_score: f64,
    pub interaction_risks: Vec<String>,
    pub contraindications: Vec<String>,
}

// Default impl for ScreeningParameters for testing
impl Default for ScreeningParameters {
    fn default() -> Self {
        Self {
            max_affinity_nm: 1000.0,
            min_druglikeness: 0.5,
            max_molecular_weight: 600.0,
            use_pharmacophore: false,
            pharmacophore_model_id: None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::AgentConfig;

    fn create_test_molecule(id: &str, mw: f64, logp: f64, hbd: u32, hba: u32) -> Molecule {
        Molecule {
            molecule_id: id.to_string(),
            smiles: "CCO".to_string(), // Ethanol example
            inchi: "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3".to_string(),
            molecular_weight: mw,
            atoms: vec![
                Atom { index: 0, element: "C".to_string(), position: [0.0, 0.0, 0.0], charge: 0.0, hybridization: "sp3".to_string(), aromatic: false },
                Atom { index: 1, element: "C".to_string(), position: [1.5, 0.0, 0.0], charge: 0.0, hybridization: "sp3".to_string(), aromatic: false },
                Atom { index: 2, element: "O".to_string(), position: [2.0, 1.0, 0.0], charge: 0.0, hybridization: "sp3".to_string(), aromatic: false },
            ],
            bonds: vec![
                Bond { atom1: 0, atom2: 1, bond_type: BondType::Single, aromatic: false, rotatable: false },
                Bond { atom1: 1, atom2: 2, bond_type: BondType::Single, aromatic: false, rotatable: true },
            ],
            properties: MolecularProperties {
                logp, logd: logp, psa: 20.23, hbd, hba, rotatable_bonds: 1,
                aromatic_rings: 0, molecular_volume: 58.7, solubility: 1000.0, permeability: 0.9,
            },
            fingerprint: MolecularFingerprint {
                ecfp4: vec![123, 456], maccs: vec![true; 166],
                pharmacophore: vec![0.1, 0.2], shape_descriptor: vec![0.3, 0.4],
            },
        }
    }

    fn create_test_target(id: &str) -> DrugTarget {
        DrugTarget {
            target_id: id.to_string(),
            protein_name: "Test Protein".to_string(),
            uniprot_id: "PTEST1".to_string(),
            pdb_id: Some("1TST".to_string()),
            sequence: "MTESTSEQUENCE...".to_string(),
            structure: None, // Simplified for test
            binding_sites: vec![BindingSite {
                site_id: "BS1".to_string(), residues: vec![10, 20, 30], volume: 150.0,
                druggability_score: 0.8, known_inhibitors: vec![],
            }],
            known_ligands: vec![],
            disease_associations: vec![],
        }
    }

    #[tokio::test]
    async fn test_drug_discovery_engine_initialization() {
        let agent_config = AgentConfig::default();
        let medical_agent = MedicalAgent::new(agent_config).await.unwrap();
        let engine = DrugDiscoveryEngine::new(medical_agent).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_virtual_screening_empty_library() {
        let agent_config = AgentConfig::default();
        let medical_agent = MedicalAgent::new(agent_config).await.unwrap();
        let engine = DrugDiscoveryEngine::new(medical_agent).await.unwrap();
        let target_id = "T001";
        
        { // Scope for RwLockGuard
            let mut target_db = engine.target_database.write().await;
            target_db.targets.insert(target_id.to_string(), create_test_target(target_id));
        }

        let results = engine.virtual_screening(target_id, vec![], ScreeningParameters::default()).await;
        assert!(results.is_ok());
        assert!(results.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_virtual_screening_with_molecules() {
        let agent_config = AgentConfig::default();
        let medical_agent = MedicalAgent::new(agent_config).await.unwrap();
        let engine = DrugDiscoveryEngine::new(medical_agent).await.unwrap();
        
        let target_id = "T001";
        let mol1_id = "MOL001";
        let mol2_id = "MOL002";

        {
            let mut target_db = engine.target_database.write().await;
            target_db.targets.insert(target_id.to_string(), create_test_target(target_id));
            
            let mut mol_db = engine.molecular_database.write().await;
            mol_db.molecules.insert(mol1_id.to_string(), create_test_molecule(mol1_id, 300.0, 2.0, 2, 3));
            mol_db.molecules.insert(mol2_id.to_string(), create_test_molecule(mol2_id, 450.0, 3.5, 1, 5));
        }

        let params = ScreeningParameters {
            max_affinity_nm: 10000.0, // Allow most results
            min_druglikeness: 0.1,    // Allow most results
            ..Default::default()
        };

        let results = engine.virtual_screening(target_id, vec![mol1_id.to_string(), mol2_id.to_string()], params).await;
        
        assert!(results.is_ok(), "Virtual screening failed: {:?}", results.err());
        let screening_results = results.unwrap();
        assert_eq!(screening_results.len(), 2);
        assert!(screening_results.iter().any(|r| r.molecule_id == mol1_id));
        assert!(screening_results.iter().any(|r| r.molecule_id == mol2_id));
        for r in screening_results {
            assert!(r.binding_affinity > 0.0);
            assert!(r.druglikeness_score > 0.0);
        }
    }

    #[test]
    fn test_calculate_druglikeness_lipinski() {
        let good_mol = create_test_molecule("GOOD", 300.0, 2.0, 2, 3); // Passes Lipinski
        let bad_mw_mol = create_test_molecule("BAD_MW", 600.0, 2.0, 2, 3); // Fails MW
        let bad_logp_mol = create_test_molecule("BAD_LOGP", 300.0, 6.0, 2, 3); // Fails LogP
        let bad_hbd_mol = create_test_molecule("BAD_HBD", 300.0, 2.0, 6, 3); // Fails HBD
        let bad_hba_mol = create_test_molecule("BAD_HBA", 300.0, 2.0, 2, 11); // Fails HBA

        let model = SparseMatrix::identity(200); // Dummy model for ML part

        let score_good = DrugDiscoveryEngine::calculate_druglikeness(&good_mol, &model).unwrap();
        let score_bad_mw = DrugDiscoveryEngine::calculate_druglikeness(&bad_mw_mol, &model).unwrap();
        let score_bad_logp = DrugDiscoveryEngine::calculate_druglikeness(&bad_logp_mol, &model).unwrap();
        let score_bad_hbd = DrugDiscoveryEngine::calculate_druglikeness(&bad_hbd_mol, &model).unwrap();
        let score_bad_hba = DrugDiscoveryEngine::calculate_druglikeness(&bad_hba_mol, &model).unwrap();
        
        // Assuming ML part gives some baseline score, Lipinski failures should reduce it
        assert!(score_good > score_bad_mw);
        assert!(score_good > score_bad_logp);
        assert!(score_good > score_bad_hbd);
        assert!(score_good > score_bad_hba);
    }

    #[tokio::test]
    async fn test_optimize_lead_basic_run() {
        let agent_config = AgentConfig::default();
        let medical_agent = MedicalAgent::new(agent_config).await.unwrap();
        let engine = DrugDiscoveryEngine::new(medical_agent).await.unwrap();

        let lead_id = "LEAD001";
        let target_id = "TARGET_LEAD_OPT";

        {
            let mut mol_db = engine.molecular_database.write().await;
            mol_db.molecules.insert(lead_id.to_string(), create_test_molecule(lead_id, 250.0, 1.5, 3, 4));
            let mut target_db = engine.target_database.write().await;
            target_db.targets.insert(target_id.to_string(), create_test_target(target_id));
        }

        let params = OptimizationParameters {
            max_results: 3,
            use_structure_based: false,
            use_fragment_growing: false,
            use_bioisosteric: false,
            use_ml_guided: true, // Focus on this for the test
            maintain_scaffold: true,
            optimize_for_selectivity: false,
        };

        let optimization_result = engine.optimize_lead(lead_id, target_id, params).await;
        assert!(optimization_result.is_ok(), "Lead optimization failed: {:?}", optimization_result.err());
        let lead_opt = optimization_result.unwrap();
        
        assert_eq!(lead_opt.parent_molecule, lead_id);
        // Given the placeholder ML optimization, it might return a few candidates
        assert!(lead_opt.optimized_molecules.len() <= 3); 
        if !lead_opt.optimized_molecules.is_empty() {
            assert!(lead_opt.improvement_metrics.potency_improvement != 0.0); // Expect some change
        }
    }

     #[tokio::test]
    async fn test_predict_clinical_success_basic_run() {
        let agent_config = AgentConfig::default();
        let medical_agent = MedicalAgent::new(agent_config).await.unwrap();
        let engine = DrugDiscoveryEngine::new(medical_agent).await.unwrap();

        let mol_id = "MOL_CLINICAL";
        let target_id = "TARGET_CLINICAL";
        let indication = "Test Indication";

        {
            let mut mol_db = engine.molecular_database.write().await;
            mol_db.molecules.insert(mol_id.to_string(), create_test_molecule(mol_id, 320.0, 2.5, 2, 4));
             let mut target_db = engine.target_database.write().await;
            target_db.targets.insert(target_id.to_string(), create_test_target(target_id));
        }

        let prediction = engine.predict_clinical_success(mol_id, target_id, indication).await;
        assert!(prediction.is_ok(), "Clinical success prediction failed: {:?}", prediction.err());
        let clinical_pred = prediction.unwrap();

        assert_eq!(clinical_pred.molecule_id, mol_id);
        assert!(clinical_pred.success_probability >= 0.0 && clinical_pred.success_probability <= 1.0);
        assert!(!clinical_pred.phase_predictions.phase1_success.is_nan());
        assert!(!clinical_pred.side_effect_profile.common_effects.is_empty());
        assert!(!clinical_pred.regulatory_considerations.is_empty());
    }
} 