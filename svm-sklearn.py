import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Load data
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

# Split data into X and y for training and testing
X = df.drop(["target"], axis="columns")
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train using SVM
training_model = SVC()
training_model.fit(X_train, y_train)

# Test accuracy using SVM
accuracy = training_model.score(X_test, y_test)
accuracy = "{:.0%}".format(accuracy)
print("Accuracy on", len(X_test), "elements:", accuracy)


# Create a separate data frame for visualisation
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

# Plot data points
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],
            color="green", marker='+')
plt.scatter(df1['petal length (cm)'],
            df1['petal width (cm)'], color="blue", marker='.')
plt.scatter(df2['petal length (cm)'],
            df2['petal width (cm)'], color="red", marker='|')
plt.show()
