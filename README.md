# Signature Identification Using Machine Learning

An end-to-end, closed-set handwritten-signature identity classifier. It
contains a reproducible classical ML baseline (RBF SVM and Random Forest), a
small augmented TensorFlow/Keras CNN, test reports and plots, and a Streamlit
upload interface.

## Assumptions

- Each class folder represents one person; image labels are not inferred from
  filenames.
- Only genuine samples are eligible. Common forged/negative folder names are
  filtered by the dataset loader.
- The 50% probability warning is a practical heuristic, not a calibrated
  open-set rejection guarantee.
- The included generated data is strictly a software demo, not genuine
  handwriting or a meaningful performance benchmark.

## Quick start

Python 3.10 or newer is recommended. Install from the project root:

```powershell
pip install -r requirements.txt
python make_demo_data.py
python src/train_classical.py
python src/train_cnn.py
python src/evaluate.py
streamlit run app.py
```

Use the demo data to validate the complete path. For actual experiments,
follow [download_data.md](download_data.md) and place data in the expected
folder layout, or set `SIGNATURE_DATA_DIR` to a dataset root. All training and
evaluation commands must use the same dataset path. Generated and trained
artifacts are stored in `models/` and `reports/`.

## Dataset layout and preprocessing

Accepted layout:

```text
data/raw/
  writer_001/
    genuine/
      sample_a.png
      sample_b.jpg
  writer_002/
    sample.png
```

Images from folders with `forg`, `forged`, `forgery`, `forgeries`, or
`negative` in the folder name are ignored. `_forg`/`_forged` identity suffixes
are ignored as well. Per image, the same OpenCV pipeline converts to
grayscale, denoises, applies Otsu inverse thresholding (white ink on black),
crops to ink, preserves aspect ratio while padding to 128 x 128, and scales to
[0, 1].

Classical features combine HOG, aspect ratio, ink density, normalized centroid,
log-scaled Hu moments, and an LBP histogram. A `StandardScaler` is fitted only
on training examples. The CNN uses small rotations, shifts, shear and zoom;
horizontal flipping is disabled because it can change handwriting identity.
All approaches share a fixed-seed, stratified 70/15/15 train/validation/test
split.

## Generated artifacts

- `models/svm.joblib`, `models/random_forest.joblib`,
  `models/cnn.keras`, `models/scaler.joblib`, and
  `models/label_encoder.joblib`
- `reports/model_comparison.csv`, model-specific classification reports and
  confusion-matrix PNGs, and `reports/cnn_training_curves.png`
- `reports/RESULTS_GUIDE.md`, which explains metrics and common error sources

CNN and classical model files are produced by their respective training
commands. `src/evaluate.py` evaluates whichever trained models are present and
requires at least one.

## Reading results

Accuracy is the fraction of correct test predictions. Macro precision/recall/F1
give each identity equal weight; weighted scores account for the number of test
examples per identity. Confusion-matrix rows are actual people and columns are
predictions. CNN top-3 accuracy counts a result as correct when its identity
appears among the three highest-probability classes. Similar writing styles,
small or imbalanced samples, image quality, cropping and pen/pressure changes
often contribute to misclassification. Test data is held out by image, not by
session or acquisition source.

In a sample run with the bundled generator (240 images, ten classes, seed 42),
the held-out accuracy was about 86% for SVM and 97% for Random Forest. CNN
accuracy was near chance (about 8% in that run), illustrating that the tiny,
font-rendered demo is not a useful CNN benchmark. Results may vary with
available system fonts and library versions. Real-dataset accuracy cannot be
estimated from these artificial images; use the saved held-out reports for
your actual dataset.

## Limitations and next steps

This is **identification**, not signature verification: it predicts one of the
known identities. Forgery detection is not handled, and the classifier is
closed-set; the warning threshold is not a reliable unknown-person detector.
Suggested improvements:

1. Evaluate on writer-disjoint acquisition sessions and add probability
   calibration plus an open-set rejection method.
2. Add a verification/forgery task with genuine and skilled-forgery labels,
   appropriate pair/triplet learning, and security-aware evaluation.
3. Expand genuine samples and compare writer-independent CNN/metric-learning
   approaches with validated preprocessing and augmentation.
