//! Clinical validation and metrics module

use crate::Result;

/// Clinical metrics for medical AI validation
#[derive(Debug, Clone)]
pub struct ClinicalMetrics {
    pub dice_coefficient: f64,
    pub hausdorff_distance: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub accuracy: f64,
}

impl ClinicalMetrics {
    /// Create new clinical metrics
    pub fn new() -> Self {
        Self {
            dice_coefficient: 0.0,
            hausdorff_distance: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            accuracy: 0.0,
        }
    }

    /// Compute metrics from predictions and ground truth
    pub fn compute(predictions: &[f64], ground_truth: &[f64], threshold: f64) -> Result<Self> {
        if predictions.len() != ground_truth.len() {
            return Err("Predictions and ground truth must have same length".into());
        }

        let mut tp = 0.0;
        let mut tn = 0.0;
        let mut fp = 0.0;
        let mut fn_count = 0.0;

        for (&pred, &truth) in predictions.iter().zip(ground_truth.iter()) {
            let pred_binary = if pred >= threshold { 1.0 } else { 0.0 };
            let truth_binary = if truth >= threshold { 1.0 } else { 0.0 };

            match (pred_binary, truth_binary) {
                (1.0, 1.0) => tp += 1.0,
                (0.0, 0.0) => tn += 1.0,
                (1.0, 0.0) => fp += 1.0,
                (0.0, 1.0) => fn_count += 1.0,
                _ => {}
            }
        }

        let sensitivity = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };
        let accuracy = (tp + tn) / (tp + tn + fp + fn_count);
        let dice_coefficient = if 2.0 * tp + fp + fn_count > 0.0 { 
            2.0 * tp / (2.0 * tp + fp + fn_count) 
        } else { 
            0.0 
        };

        Ok(Self {
            dice_coefficient,
            hausdorff_distance: 0.0, // Placeholder - would need actual implementation
            sensitivity,
            specificity,
            accuracy,
        })
    }

    /// Check if metrics meet clinical thresholds
    pub fn meets_clinical_threshold(&self) -> bool {
        self.dice_coefficient > 0.8 &&
        self.sensitivity > 0.85 &&
        self.specificity > 0.85 &&
        self.accuracy > 0.9
    }
}

impl Default for ClinicalMetrics {
    fn default() -> Self {
        Self::new()
    }
} 