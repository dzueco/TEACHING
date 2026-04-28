# SVM Iris Example

This folder contains the material used in class for a first practical example of
support vector machines with the Iris data set.

## Files

- `environment.yml`: conda environment with the Python packages used by the example.
- `svm/svm_iris_pythonic.py`: Python script that loads the Iris data set, trains an
  SVM classifier, evaluates it, and creates PCA-based figures.
- `data/iris.data`: Iris data file used by the script.
- `figures/`: suggested folder for generated plots.

## Create The Conda Environment

From this `2025-2026` folder, create the environment with:

```bash
conda env create -f environment.yml
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
```

Then activate it:

```bash
conda activate svm
```

## Run The Script

From this `2025-2026` folder:

```bash
python svm/svm_iris_pythonic.py
```

The script prints the train/test shapes, the accuracy, the confusion matrix, and
the classification report. It also opens two matplotlib windows:

- a PCA preview of the raw Iris data;
- a PCA decision-boundary visualization of the SVM classification problem.

## Save The Figures

To save both figures and also display them:

```bash
python svm/svm_iris_pythonic.py \
  --save-preview-plot figures/iris_pca_preview.png \
  --save-plot figures/iris_pca_boundary.png
```

To save the figures without opening matplotlib windows:

```bash
python svm/svm_iris_pythonic.py \
  --save-preview-plot figures/iris_pca_preview.png \
  --save-plot figures/iris_pca_boundary.png \
  --no-show
```

The `--save-preview-plot` option saves the raw PCA preview. The `--save-plot`
option saves the PCA decision-boundary figure. Both options need a filename.

## Notes

The classifier is trained on the original four Iris measurements using an RBF
kernel. The 2D figures use PCA so that the data and a decision boundary can be
shown on a page or slide. The visualization is pedagogical; it is not the exact
four-dimensional geometry of the trained RBF model.
