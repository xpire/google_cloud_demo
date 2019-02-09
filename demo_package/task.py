# Justin Or, Feb 2019

# task.py for use on Google CloudML
# Contains all the main logic of the package.

# Constants will go in the __init__.py file

# Usage: python -m demo.task (from outside this directory)
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python task.py

from . import * 
from .packagedModel import *
from .model_API import *
import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # Turn on tensorboard:
    #   tensorboard --logdir={$MODEL_DIR}
    features = build_model_columns()
    train_spec = tf.estimator.TrainSpec(input_fn=lambda : input_train_set(), max_steps=NUMITERATIONS)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda : input_eval_set())
    print(train_spec)
    print(eval_spec)
    mdl = Model() \
        .set_feat_col(features) \
        .set_model()

    mdl.set_train_spec(train_spec)
    mdl.set_eval_spec(eval_spec)
    mdl.train_and_evaluate()
    mdl.evaluate()