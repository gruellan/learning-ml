import pandas as pd
from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = data('Pima.tr')

# Test train split for supervised learning (randomly splits data for us)
X_train, X_test, y_train, y_test = train_test_split(dataset.skin, dataset.bmi)

# Create model
lr = LinearRegression()
# Feed data to model (X is reshaped to make it a 2D vector)
lr.fit(X_train.values.reshape(-1,1), y_train.values)

# Predict on test data 
prediction = lr.predict(X_test.values.reshape(-1,1))

# Get R^2 of the model (varies each run due to random test train split)
print("R^2 Score:", lr.score(X_test.values.reshape(-1,1), y_test.values))

# Plot LR line and 
plt.plot(X_test, prediction, label="Linear Regression Line", color="b")
plt.scatter(X_test, y_test, label="Data Points", color="g", alpha=.7)
plt.title("Predicting BMI based on tricep skin fold measurement")
plt.show()