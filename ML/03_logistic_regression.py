# Experiment 3: Logistic Regression

import numpy as np
from sklearn.linear_model import LogisticRegression

# Input data: hours studied
X = np.array([[1], [2], [3], [4], [5], [6]])

# Output: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict for a student who studied 3.5 hours
new_student = np.array([[3.5]])

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("Predicted class:", prediction[0])
print("Probability of fail:", probability[0][0])
print("Probability of pass:", probability[0][1])

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")
