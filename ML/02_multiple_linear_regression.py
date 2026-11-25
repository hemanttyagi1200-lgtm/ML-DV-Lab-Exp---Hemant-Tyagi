# Experiment 2: Multiple Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression

# Input data: hours studied and hours slept
X = np.array([
    [2, 6],
    [3, 7],
    [4, 6],
    [5, 8],
    [6, 7]
])

# Output data: marks obtained
y = np.array([50, 60, 65, 75, 80])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for a new student
new_student = np.array([[4, 7]])
prediction = model.predict(new_student)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted marks:", prediction[0])
