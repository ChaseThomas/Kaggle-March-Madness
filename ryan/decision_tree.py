from sklearn import tree as t
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np


def train_tree(x_data, y_data, seed=None):
    the_tree = t.DecisionTreeRegressor(random_state=seed)
    return the_tree.fit(x_data, y_data)


def predict(tree, x_data):
    return tree.predict(x_data)


def test_accuracy(tree, x_data, y_data):
    predictions = np.round(predict(tree, x_data))
    return np.mean(np.abs(predictions-y_data))
