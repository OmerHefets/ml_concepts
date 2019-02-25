import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)


def split_train_val_test(X, y, val_size, test_size, random_state):
    val_size = val_size / (1 - test_size)  # make val_size equal the relative size from X
    X_trainAndVal, X_test, y_trainAndVal, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_trainAndVal, y_trainAndVal, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.transform(data)
    mean = scaler.mean_
    std = scaler.var_
    return mean, std, scaled_data


X, y = boston.data, boston.target
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, val_size=0.2, test_size=0.2, random_state=42)
