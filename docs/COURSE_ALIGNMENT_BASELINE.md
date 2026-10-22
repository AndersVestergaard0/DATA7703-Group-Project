# DATA7703 course alignment for the PCA/linear-SVM baseline

2026-10-22. The local inventory contains 70 PDF/notebook/Python files, including duplicate copies. All available PDFs were text-extracted for method lookup; relevant practical code was inspected. This is an inventory of current local material, not a claim that every course week is complete.

| Implementation | Verified course source | What is reused |
|---|---|---|
| PCA centring and projection | Week4/Lab_and_Prac/Prac 4 solns.ipynb, zero-based cells 5,8,11,17,19 | Training mean, covariance/principal directions, projection |
| Image vectors/PCA | Week4/Lecture_Slides/DATA7703_Lec4.pdf, image examples and principal-component material; text matches pages 21,23,24,26 | Pixel representation and lower-dimensional projection |
| Linear SVM and C | Week6/Lab_and_Prac/Prac Week 6 solutions.ipynb, zero-based cell 2; parameter comparisons in subsequent cells | sklearn.svm.SVC(kernel='linear',C=...) |
| Confusion matrix | Same Week6 practical, zero-based cell 36 | Separate true/predicted labels and error counts |
| Evaluation, reproducibility, error analysis | Official DATA7703 Group Project 2026 brief, pages 1–2 | Metrics, configurations, findings and limitations |

sklearn PCA with randomized SVD is a numerical implementation choice; the course practical illustrates the same concept with covariance/eigendecomposition. 32x32 inputs, 20/50 components, C=0.1/1.0, F1-based validation selection and seed 42 are declared experiment settings, not settings mandated by the lecturer. RGB/padding/file handling are dataset engineering steps. No HOG, pretrained image features, nonlinear kernel, ensemble or new model family was introduced into Felix's baseline.

Local Week10–Week12 folders contain no PDF/notebook/Python material in this inventory. The baseline depends on verified Week4/Week6 material; do not attribute Grad-CAM implementation or clinical conclusions to missing lecture material. A complete local material inventory was retained during development; course files are not needed at runtime. Notebook cells remain runnable and readable for personal learning.
