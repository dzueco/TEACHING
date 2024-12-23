import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np


# =======================
# load data
# =======================
def load_dataset(dataset_name):
    """
    Load and preprocess the dataset based on its name.

    Args:
        dataset_name (str): The name of the dataset (e.g., "iris", "wdbc").

    Returns:
        tuple: Features (X) and labels (Y).
    """
    if dataset_name == "iris":
        column_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]
        data = pd.read_csv("../data/iris.data", names=column_names)
        X = data.drop(columns=["class"])
        Y = (
            data["class"]
            .replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
            .astype(int)
        )

    elif dataset_name == "wdbc":
        column_names = [
            "ID",
            "Diagnosis",
            "Radius_mean",
            "Texture_mean",
            "Perimeter_mean",
            "Area_mean",
            "Smoothness_mean",
            "Compactness_mean",
            "Concavity_mean",
            "Concave_points_mean",
            "Symmetry_mean",
            "Fractal_dimension_mean",
            "Radius_se",
            "Texture_se",
            "Perimeter_se",
            "Area_se",
            "Smoothness_se",
            "Compactness_se",
            "Concavity_se",
            "Concave_points_se",
            "Symmetry_se",
            "Fractal_dimension_se",
            "Radius_worst",
            "Texture_worst",
            "Perimeter_worst",
            "Area_worst",
            "Smoothness_worst",
            "Compactness_worst",
            "Concavity_worst",
            "Concave_points_worst",
            "Symmetry_worst",
            "Fractal_dimension_worst",
        ]
        data = pd.read_csv("../data/wdbc.data", names=column_names)
        X = data.drop(columns=["ID", "Diagnosis"])
        Y = data["Diagnosis"].replace({"B": 0, "M": 1}).astype(int)

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

    return X.values, Y.values


# =======================
# data_split
# =======================
def data_split(X, Y, holdout_size=0.02, test_size=0.2, random_state=1):
    """
    Split the dataset into training, testing, and holdout sets.

    Args:
        X (np.ndarray): The feature matrix.
        Y (np.ndarray): The target labels.
        holdout_size (float, optional): Proportion of the data to reserve as the holdout set.
                                        Default is 0.02 (2% of the dataset).
        test_size (float, optional): Proportion of the remaining data (after the holdout split)
                                     to reserve as the test set. Default is 0.2 (20%).
        random_state (int, optional): Random seed for reproducibility. Default is 1.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - Y_train (np.ndarray): Training labels.
            - X_test (np.ndarray): Testing features.
            - Y_test (np.ndarray): Testing labels.
            - X_holdout (np.ndarray): Holdout features
            - Y_holdout (np.ndarray): Holdout labels

    Notes:
        - The holdout set is determined by `holdout_size`.
        - The test set is determined by `test_size`, applied to the remaining data after the holdout split.
        - The splits are reproducible due to the fixed `random_state`.
    """

    X_train, X_holdout, Y_train, Y_holdout = train_test_split(
        X, Y, test_size=holdout_size, random_state=random_state
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=random_state
    )

    return X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout


# =======================
# visualize raw data (seaborn)
# =======================
def visualize_data(X, Y, column_names):
    """
    Visualize the data using Seaborn pairplot.

    Args:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Labels.
        column_names (list): List of feature names for X.
    """
    data = pd.DataFrame(X, columns=column_names)
    data["class"] = Y
    sns.pairplot(data, hue="class")
    plt.show()


# =======================
# visualize raw data (using pca)
# =======================
def visualize_with_pca(X, Y):
    """
    Reduces the dimensionality of the dataset using PCA and visualizes the first two principal components.  It then creates a scatter plot of the first two principal components, with points colored
    according to their class labels in `Y`.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features). The data to be reduced.
        Y (np.ndarray): Labels or class information for each sample in `X`. Should be of shape (n_samples,).

    Returns:
        None: The function directly shows the scatter plot of the first two principal components.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    data_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    data_pca["class"] = Y
    sns.scatterplot(data=data_pca, x="PC1", y="PC2", hue="class")
    plt.show()


# ======================
# sort indexes for Kernel plotting
# =======================


def sort_K(K, Y_train):
    """_summary_

    Args:
        K (_type_): _description_
        Y (_type_, optional): _description_. Defaults to Y_train.

    Returns:
        _type_: _description_
    """
    sorted_indices = np.argsort(Y_train)
    unique, counts = np.unique(Y_train, return_counts=True)
    print("classes:", unique)
    print("# elemements per class", counts)
    K_sorted = K[sorted_indices, :][:, sorted_indices]
    return K_sorted


# =======================
# plot kernel matrix (sorted by labels)
# =======================
def plot_kernel_matrix(K, title="Kernel Matrix"):
    """
    Plot a heatmap of the Kernel matrix.

    Args:
        K (np.ndarray): Kernel matrix of shape (n_samples, n_samples).
                        Each element represents the similarity between two samples.
        title (str, optional): Title for the plot. Default is 'Kernel Matrix'.

    Returns:
        None: Displays a heatmap of the Kernel matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(K, cmap="coolwarm", annot=False, square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.show()
