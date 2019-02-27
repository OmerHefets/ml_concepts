import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()


def split_train_val_test(X, y, val_size, test_size, random):
    val_size = val_size / (1 - test_size)  # make val_size equal the relative size from X
    X_trainNVal, X_test, y_trainNVal, y_test = train_test_split(X, y, test_size=test_size, random_state=random)
    X_train, X_val, y_train, y_val = train_test_split(X_trainNVal, y_trainNVal, test_size=val_size, random_state=random)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    mean = scaler.mean_
    std = scaler.var_
    return mean, std, scaled_data


def learning_curve(X_train, y_train, X_val, y_val, model):
    train_error, val_error = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        predictions_train = model.predict(X_train[:m])
        predictions_validation = model.predict(X_val)
        train_error.append(mean_squared_error(y_train[:m], predictions_train))
        val_error.append(mean_squared_error(y_val, predictions_validation))
    plt.plot(train_error, "r", label="train")
    plt.plot(val_error, "b", label="val")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.axis([0, 303, 0, 100])
    plt.show()


X, y = boston.data, boston.target
X_train_boston, X_val_boston, X_test_boston, y_train_boston, y_val_boston, y_test_boston = \
    split_train_val_test(X, y, val_size=0.2, test_size=0.2, random=42)

boston_mean, boston_std, X_train_boston_scaled = scale_data(X_train_boston)
_, _, X_val_boston_scaled = scale_data(X_val_boston)

lin_reg = LinearRegression()
learning_curve(X_train_boston_scaled, y_train_boston, X_val_boston_scaled, y_val_boston, lin_reg)
