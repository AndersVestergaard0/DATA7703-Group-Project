# PCA + linear SVM validation snapshot

Run date: 22 October 2026. Positive class: Parasitized=1; Uninfected=0.

The unchanged patient manifest contains 19,124 training, 4,324 validation and 4,110 held-out test images. The baseline uses 32x32 RGB pixels, PCA fitted on training images only, and a linear SVC. The declared candidates were 20/50 components and C=0.1/1.0, selected by validation F1 then recall and smaller dimension/C. Selected: 20 components, C=0.1.

| Validation metric | Value |
|---|---:|
| Accuracy | 0.658418 |
| Precision | 0.666385 |
| Recall | 0.696113 |
| F1 | 0.680925 |
| ROC-AUC | 0.697459 |
| TN / FP / FN / TP | 1271 / 789 / 688 / 1576 |

These are validation-selected results. No test evaluation was performed. The four candidates, prediction rows, patient-level error counts, configuration and figures are included in this directory.

At this run's base commit, the shared preprocessing module was still a stub. The explicit fallback matches the current CNN notebook's RGB, bilinear resize, centred black padding and [0,1] scaling, using 32x32 inputs. Recheck pixel equivalence and validation results after integrating the shared implementation.

The source notebook is published without cached outputs; its full executed copies and serialized model were retained locally. Run the notebook or scripts/run_baseline_notebook.py to regenerate outputs and the model, then scripts/verify_baseline.py to verify predictions. Set MALARIA_DATA_ROOT first, as described in the repository README. SVM margins are decision scores, not calibrated probabilities.
