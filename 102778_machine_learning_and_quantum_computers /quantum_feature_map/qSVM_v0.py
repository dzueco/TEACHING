# %%
import os
import sys
import numpy as np

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml

""" add the current folder for importing modules """
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload

import moduqusvm as mdqsvm
import modudata as mddt

mdqsvm = reload(mdqsvm)
mddt = reload(mddt)
from moduqusvm import svm_models


# %%

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

data = pd.read_csv("../data/iris.data", names=column_names)

# check data size and the number of features
print(data.shape)

X = data.drop(columns=["class"])
Y = (
    data["class"]
    .replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    .astype(int)
)

X = X.values
Y = Y.values


mddt.visualize_with_pca(X, Y)


X_train, Y_train, X_test, Y_test, X_holdout, Y_holdout = mddt.data_split(X, Y)


# %%

# "classical" vsm
selected_kernel = "linear"
clf = svm_models[selected_kernel]["model"]

# train:
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(f"Precisión: {accuracy}")

Y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("confusion matrix:")
print(conf_matrix)

# visualize kernels
K_train = svm_models[selected_kernel]["kernel_matrix"](X_train)

K_sorted = mddt.sort_K(K_train, Y_train)

mddt.plot_kernel_matrix(K_sorted, title=selected_kernel)

# %%
# quantum svm
num_qubits = 4

# visualize quantum feature map
x = [0.1, 0.4, 0.6, 0.2]
mdqsvm.cir_ej(4, lambda: mdqsvm.ansatz(x, 4))

adjoint_ansatz = qml.adjoint(mdqsvm.layer)(x, 4)
mdqsvm.cir_ej(4, lambda: adjoint_ansatz)
# %%
selected_kernel = "quantum"
clf = svm_models[selected_kernel]["model"](num_qubits)

clf.fit(X_train, Y_train)


accuracy = clf.score(X_test, Y_test)
print(f"Precisión: {accuracy}")

Y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("confusion matrix:")
print(conf_matrix)


# %%
# visualize kernels

K_train = svm_models[selected_kernel]["kernel_matrix"](X_train, num_qubits)
K_sorted = mddt.sort_K(K_train, Y_train)
mddt.plot_kernel_matrix(K_sorted, title=selected_kernel)

# %%
