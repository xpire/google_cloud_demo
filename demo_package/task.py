# Justin Or, Feb 2019

# task.py for use on Google CloudML
# Contains all the main logic of the package.

# Constants will go in the __init__.py file

# Usage: python -m demo.task (from outside this directory)
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python task.py

# Usage with gcloud ml-engine:
# gcloud ml-engine jobs submit training TESTJOB7 \
# --runtime-version=1.11 \
# --python-version 3.5 \
# --region=us-central1 \
# --module-name=demo_package.task \
# --job-dir=gs://rossmann-cbd/output/ \
# --scale-tier=BASIC  \
# --package-path=demo_package \
# -- --train_data_paths="gs://rossmann-cbd/trainPartitioned.csv" \
# --eval_data_paths="gs://rossmann-cbd/testPartitioned.csv" \
# --output_dir="model_check"

# non condensed: 
# gcloud ml-engine jobs submit training TRIALJOB2 --runtime-version=1.11 --python-version 3.5 --region=us-central1 --module-name=demo_package.task --job-dir=gs://rossmann-cbd/output/ --scale-tier=BASIC  --package-path=demo_package -- --train_data_paths="gs://rossmann-cbd/trainPartitioned.csv" --eval_data_paths="gs://rossmann-cbd/testPartitioned.csv" --output_dir="model_check"

# Example of predicting with a model generated by this package:
# gcloud ml-engine predict --model rossmann_cbd_test_7 --version rossmann_cbd_test_7 --json-instances sample_request.json
# sample_request.json has:
# {"CompetitionDistance": 1200, "Year": 2015, "Promo2": 0, "Assortment": "c", "SchoolHoliday": 0, "StateHoliday": "0", "DayOfWeek": 0, "StoreType": "c", "Promo": 1, "WeekOfYear": 27, "Month": 5, "Open": 1}
# Output:
# PREDICTIONS
# [7051.439453125]

from . import * 
from .packagedModel import *
from .model_API import *
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # Turn on tensorboard:
    #   tensorboard --logdir={$MODEL_DIR}
    features = build_model_columns()
    mdl = Model() \
        .set_feat_col(features) \
        .set_model()

    mdl.set_train_spec()
    mdl.set_eval_spec()
    mdl.train_and_evaluate()
    mdl.evaluate()