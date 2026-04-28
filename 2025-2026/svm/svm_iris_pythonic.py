from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "iris.data"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "class"
COLUMN_NAMES = FEATURE_COLUMNS + [TARGET_COLUMN]

CLASS_TO_NUMBER = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}
CLASS_NAMES = list(CLASS_TO_NUMBER)

# for reproducibility, we use in train_test_split function used in our function split_data
RANDOM_STATE = 1


def load_iris_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the Iris CSV file and return features and numeric labels."""
    # Read.  Assign the column names.
    data = pd.read_csv(data_path, names=COLUMN_NAMES)

    # Keep only the numeric feature columns 
    features = data[FEATURE_COLUMNS].to_numpy()
    # Convert each flower name into the integer label expected by scikit-learn.
    labels = data[TARGET_COLUMN].map(CLASS_TO_NUMBER)

    # Check whether the file contains class names that are not in our mapping.
    # pandas methods: isna() check if there is a Nan, any() checks if there at least one True
    if labels.isna().any():
        # Collect the unexpected labels so the error message is explicit.
        # pandas object unique: here the wrong class stored, but not all the error examples
        unknown_labels = sorted(data.loc[labels.isna(), TARGET_COLUMN].unique())
        # Stop immediately because we cannot train with unknown classes.
        raise ValueError(f"Unknown Iris labels found: {unknown_labels}")

   
    return features, labels.to_numpy(dtype=int)


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into training and holdout/test sets. stratifying following the labels"""
    return train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def train_svm(features_train: np.ndarray, labels_train: np.ndarray) -> SVC:
    """Train a support vector machine with an RBF kernel."""
    
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(features_train, labels_train)
    return model


def evaluate_model(
    model: SVC,
    features_test: np.ndarray,
    labels_test: np.ndarray,
) -> None:
    """Print simple classification metrics on the holdout/test set."""
    predictions = model.predict(features_test)
    accuracy = accuracy_score(labels_test, predictions)

    print(f"Accuracy: {accuracy:.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(labels_test, predictions))
    print("\nClassification report:")
    print(classification_report(labels_test, predictions, target_names=CLASS_NAMES))


def project_to_pca_2d(
    reference_features: np.ndarray,
    *feature_sets: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Fit PCA on reference data and project one or more feature sets to 2D."""
    pca = PCA(n_components=2)
    pca.fit(reference_features)
    return tuple(pca.transform(feature_set) for feature_set in feature_sets)


def visualization(
    features_2d: np.ndarray,
    labels: np.ndarray,
    figure_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Plot the raw Iris data after a PCA reduction to two dimensions."""
    plt.figure()
    plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap=plt.cm.coolwarm,
        edgecolor="k",
    )
    plt.title("Iris data after PCA reduction")
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    plt.tight_layout()

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pca_decision_boundary(
    features_train_2d: np.ndarray,
    labels_train: np.ndarray,
    features_test_2d: np.ndarray,
    labels_test: np.ndarray,
    figure_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Plot an SVM decision boundary in the 2D PCA projection."""
    model = SVC(kernel="linear", C=1.0, gamma="scale")
    model.fit(features_train_2d, labels_train)

    x_min, x_max = features_train_2d[:, 0].min() - 1, features_train_2d[:, 0].max() + 1
    y_min, y_max = features_train_2d[:, 1].min() - 1, features_train_2d[:, 1].max() + 1

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
    grid_predictions = model.predict(grid_points).reshape(x_grid.shape)

    plt.figure()
    plt.contourf(x_grid, y_grid, grid_predictions, alpha=0.35, cmap=plt.cm.coolwarm)
    plt.scatter(
        features_train_2d[:, 0],
        features_train_2d[:, 1],
        c=labels_train,
        cmap=plt.cm.coolwarm,
        edgecolor="k",
        label="train",
    )
    plt.scatter(
        features_test_2d[:, 0],
        features_test_2d[:, 1],
        c=labels_test,
        cmap=plt.cm.coolwarm,
        marker="x",
        label="test",
        linewidths=1.5,
    )
    plt.title("SVM decision boundary after PCA")
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    plt.legend()
    plt.tight_layout()

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train and visualize an SVM on the Iris data set.")
    parser.add_argument(
        "--save-preview-plot",
        type=Path,
        default=None,
        help="Optional path where the raw PCA preview figure is saved.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path where the PCA decision-boundary figure is saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features, labels = load_iris_data(DATA_PATH)
    features_train, features_test, labels_train, labels_test = split_data(
        features,
        labels,
    )

    print(f"Data shape: {features.shape}")
    print(f"Train shape: {features_train.shape}")
    print(f"Test shape: {features_test.shape}\n")

    features_2d, features_train_2d, features_test_2d = project_to_pca_2d(
        features_train,
        features,
        features_train,
        features_test,
    )
    visualization(
        features_2d,
        labels,
        figure_path=args.save_preview_plot,
        show_plot=not args.no_show,
    )

    model = train_svm(features_train, labels_train)
    evaluate_model(model, features_test, labels_test)
    plot_pca_decision_boundary(
        features_train_2d,
        labels_train,
        features_test_2d,
        labels_test,
        figure_path=args.save_plot,
        show_plot=not args.no_show,
    )


# to prevent run it if imported.
if __name__ == "__main__":
    main()
