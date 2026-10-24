# Frozen PCA + linear SVM: final test
The original validation-selected checkpoint (20 PCA components; linear SVC C=0.1; RGB 32x32) was evaluated on the unchanged patient-level test manifest. No PCA or SVM fit occurred during test evaluation.

Test: 4,110 images /31 recorded patient ID groups; Parasitized=1.
Accuracy 65.67%, precision 65.11%, recall 60.90%, F1=0.6294, ROC-AUC=0.7133.
TN 1501, FP 642, FN 769, TP 1198. The prior validation predictions replayed unchanged; the checkpoint file hash stayed unchanged. Validation recall was 69.61%, so report lower test recall even though overall accuracy is close.

## Reproduce or verify
Python 3.11 with requirements-baseline-frozen.txt. Set MALARIA_DATA_ROOT to the directory containing Parasitized/ and Uninfected/.

    python scripts/evaluate_frozen_svm.py --confirm-frozen-protocol
    python scripts/verify_frozen_svm_test.py

Evaluation checks checkpoint/split/preprocessing hashes and reads completed results without repeating inference. The verifier recomputes metrics from CSVs, needs no images and performs no inference. Notebook 07 reads the deliverable; Notebook 02 is the validation-selection workflow. The frozen checkpoint and metadata are under models/.

checkpoint_info.json records versions, hashes and validation replay. test_predictions.csv retains filename/patient/true label/prediction/signed decision score/outcome; scores are not probabilities. evaluation.png shows confusion/ROC/PR; error_cases.png samples TP/TN/FP/FN with a fixed seed. patient_test_errors.csv includes image counts and error rates. model_comparison.csv/png compares all pipelines on the same test cohort. verification.json records independent checks.

CNN/ResNet source predictions and metrics are from artifacts/test_evaluation/. Their inputs are 128px; ResNet uses ImageNet normalization and pretrained initialization. This compares complete pipelines on one split, not an isolated architecture experiment or a significance test. Error images and Grad-CAM do not establish clinical safety or validated parasite localization.
