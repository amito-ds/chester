# import necessary libraries
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# load mnist dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# instantiate and fit the model
model = CatBoostClassifier(
    task_type="CPU", loss_function="MultiClass", n_estimators=500
)

model.fit(X_train, y_train)

# report accuracy on the validation set
print("Accuracy score on validation set: {:0.2f}".format(model.score(X_val, y_val)))
