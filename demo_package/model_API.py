"""
Simple wrapper API for choosing different models 
{(maybe) improve for later escaping}
                _ ___                /^^\ /^\  /^^\_
    _          _@)@) \            ,,/ '` ~ `'~~ ', `\.
  _/o\_ _ _ _/~`.`...'~\        ./~~..,'`','',.,' '  ~:
 / `,'.~,~.~  .   , . , ~|,   ,/ .,' , ,. .. ,,.   `,  ~\_
( ' _' _ '_` _  '  .    , `\_/ .' ..' '  `  `   `..  `,   \_
 ~V~ V~ V~ V~ ~\ `   ' .  '    , ' .,.,''`.,.''`.,.``. ',   \_
  _/\ /\ /\ /\_/, . ' ,   `_/~\_ .' .,. ,, , _/~\_ `. `. '.,  \_
 < ~ ~ '~`'~'`, .,  .   `_: ::: \_ '      `_/ ::: \_ `.,' . ',  \_
  \ ' `_  '`_    _    ',/ _::_::_ \ _    _/ _::_::_ \   `.,'.,`., \-,-,-,_,_,
   `'~~ `'~~ `'~~ `'~~  \(_)(_)(_)/  `~~' \(_)(_)(_)/ ~'`\_.._,._,'_;_;_;_;_;

    Author: Even Tang
    Project: project Rossman
    Date: Jan 2019

"""

"""
Importing libraries
"""
from . import *
from .packagedModel import *
from .task import *
import tensorflow as tf
import numpy as np
import pandas as pd

# Model for fixed model
class Model:
    def __init__(self):
        # currently what type of optimizer ?
        self._feature_col = None 
        self._model = None
        self._train_spec = None
        self._eval_spec = None

    # To implement in the future: User only needs to enter:
    # - Date:
            # Year
            # DayOfWeek
            # Month
            # MonthWeekOfYear
    # - StoreID:
            # CompetitionDistance
            # Promo2
            # Assortment
            # Storetype
    # - Open
    # - Promo2
    # - StateHoliday
    # - SchoolHoliday
    # this change will increase usability
    def serving_input_function(self):
        feature_placeholders = {}
        for col in integer_features:
            feature_placeholders[col] = tf.placeholder(tf.int64, [None])
        for col in boolean_features:
            feature_placeholders[col] = tf.placeholder(tf.int64, [None])
        for key_name in categorical_features.keys():
            feature_placeholders[key_name] = tf.placeholder(tf.string, [None])
        for key_name in categorical_identity_features.keys():
            feature_placeholders[key_name] = tf.placeholder(tf.int64, [None])
        for key_name in bucket_categorical_features.keys():
            feature_placeholders[key_name] = tf.placeholder(tf.int64, [None])

        features = { key_name: tf.expand_dims(tensor, -1) for key_name, tensor in feature_placeholders.items() }

        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

    # Set feature column of the model
    def set_feat_col(self, feature_col):
        # Checking function
        if (feature_col == None): 
            print("Model type error.")
        else: 
            self._feature_col = feature_col
        return self 

    # Load the model
    def set_model(self): 
        if (self._feature_col == None):
            print("Model error.")
        else:
            training_config = tf.estimator.RunConfig(model_dir=PATH, save_summary_steps=100, save_checkpoints_steps=100, save_checkpoints_secs=120)
            # Set the correct model
            # self._model = tf.estimator.DNNRegressor(
            #     hidden_units=[20],
            self._model = tf.estimator.LinearRegressor(
                config=training_config,
                feature_columns=self._feature_col,
                # model_dir=PATH,
                optimizer=tf.train.AdamOptimizer(learning_rate=LEARNINGRATE)
                # warm_start_from=PATH
            )

        return self

    

    # define Train Spec
    def set_train_spec(self):
        train_spec = tf.estimator.TrainSpec(input_fn=lambda : input_train_set(), max_steps=NUMITERATIONS)
        self._train_spec = train_spec
        
    # define Eval Spec
    def set_eval_spec(self):
        latest_exporter  = tf.estimator.LatestExporter(
            name="models",
            serving_input_receiver_fn=self.serving_input_function,
            exports_to_keep=10
        )
        best_exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=self.serving_input_function,
            exports_to_keep=1
        )
        exporters = [latest_exporter, best_exporter]
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda : input_eval_set(),
            exporters=exporters,
            throttle_secs=600
        )
        self._eval_spec = eval_spec

    # Train and Evaluate function
    def train_and_evaluate(self):
        if self._eval_spec == None or self._train_spec == None:
            print("Please instantiate eval and train specs.")
        else: 
            tf.estimator.train_and_evaluate(self._model, self._train_spec, self._eval_spec)

    def evaluate(self):
        self._model.evaluate(input_fn=lambda: input_eval_set(), steps=10)      

