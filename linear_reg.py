import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

# random data
np.random.seed(40)
X = 2 * np.random.rand(200, 1)
np.random.seed(4)
y = 5 + 4 * X + np.random.randn(200, 1)

# Making polynomial features
polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_polynomial = polynomial.fit_transform(X)

# Using Normal Equation
X_b = np.c_[np.ones((200, 1)), X]
theta_ne = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
predictions_ne = X_b.dot(theta_ne)

# Using sk_learn's Normal Equation
lin_reg = LinearRegression()
lin_reg.fit(X, y)
theta_sk = np.array([lin_reg.intercept_, lin_reg.coef_])
predictions_sk = X_b.dot(theta_sk)

# Using Stochastic regression
sgd_reg = SGDRegressor(n_iter=10, penalty=None, eta0=0.1)
sgd_reg.fit(X, y)
theta_sgd = np.array([sgd_reg.intercept_, sgd_reg.coef_])
predictions_sgd = X_b.dot(theta_sgd)

# Plot the results
plt.plot(X, y, 'b.')
plt.plot(X, predictions_sk, 'g-')
plt.plot(X, predictions_ne, 'r')
plt.plot(X, predictions_sgd, 'y')
plt.axis([0, 2, 0, 15])
plt.show()
