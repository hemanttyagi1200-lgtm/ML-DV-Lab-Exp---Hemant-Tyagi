# Experiment 4: Regression Models with Regularization

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Input data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2, 4, 5, 4, 5, 7])

# Create models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)

# Train models
linear_model.fit(X, y)
ridge_model.fit(X, y)
lasso_model.fit(X, y)

# Predict for a new value
new_value = np.array([[7]])

linear_prediction = linear_model.predict(new_value)
ridge_prediction = ridge_model.predict(new_value)
lasso_prediction = lasso_model.predict(new_value)

# Display results
print("Linear Regression Prediction:", linear_prediction[0])
print("Ridge Regression Prediction:", ridge_prediction[0])
print("Lasso Regression Prediction:", lasso_prediction[0])

print("Linear Regression Coefficient:", linear_model.coef_[0])
print("Ridge Regression Coefficient:", ridge_model.coef_[0])
print("Lasso Regression Coefficient:", lasso_model.coef_[0])
