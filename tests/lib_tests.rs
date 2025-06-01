use openalgebra_medical::*;

#[test]
fn test_library_initialization() {
    assert!(init().is_ok());
}

#[test]
fn test_dicom_processor() {
    let processor = DicomProcessor::new();
    let result = processor.process_dicom_series("test_path");
    assert!(result.is_ok());
}

#[test]
fn test_medical_tensor() {
    let tensor = MedicalTensor::new(vec![10, 10, 10]);
    assert_eq!(tensor.shape(), &[10, 10, 10]);
    assert_eq!(tensor.nnz(), 0);
}

#[test]
fn test_sparse_cnn() {
    let model = SparseCNN::new()
        .anatomy("brain")
        .task("segmentation")
        .build();
    
    assert!(model.is_ok());
    let model = model.unwrap();
    assert_eq!(model.anatomy(), "brain");
    assert_eq!(model.task(), "segmentation");
    assert!(!model.is_trained());
}

#[test]
fn test_clinical_metrics() {
    let metrics = ClinicalMetrics::new();
    assert_eq!(metrics.dice_coefficient, 0.0);
    assert_eq!(metrics.accuracy, 0.0);
}

#[test]
fn test_clinical_metrics_computation() {
    let predictions = vec![0.9, 0.1, 0.8, 0.2, 0.7];
    let ground_truth = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    
    let metrics = ClinicalMetrics::compute(&predictions, &ground_truth, 0.5);
    assert!(metrics.is_ok());
    
    let metrics = metrics.unwrap();
    assert!(metrics.accuracy > 0.0);
    assert!(metrics.dice_coefficient >= 0.0);
}

#[test]
fn test_medical_tensor_from_dense() {
    let dense_data = vec![0.1, 0.0, 0.5, 0.0, 0.8, 0.0];
    let shape = vec![2, 3];
    
    let sparse_tensor = MedicalTensor::from_dense(&dense_data, shape, 0.3);
    assert!(sparse_tensor.is_ok());
    
    let tensor = sparse_tensor.unwrap();
    assert_eq!(tensor.nnz(), 2); // Only 0.5 and 0.8 are above threshold 0.3
    assert!(tensor.density() > 0.0);
} 