# Justin Or, Feb 2019

# task.py for use on Google CloudML
# Contains all the main logic of the package.

# Constants will go in the __init__.py file

# Usage: python -m demo.task (from outside this directory)
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python task.py

from . import * 
from demo.packagedModel import *
from demo.model_API import *
import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # train, store = input_data(train_file, store_file)
    # preprocess(train, store)
    print(train_file)
    dataset = input_evaluation_set(2)
    features = build_model_columns()
    mdl = Model()
    mdl.set_feat_col(features)
    mdl.set_model()
    # mdl.train()

    # classifier = tf.estimator.DNNRegressor(
    #     feature_columns=features,
    #     hidden_units=[10, 10]
    # )