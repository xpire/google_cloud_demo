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
            # Set the correct model
            # self._model = tf.estimator.DNNRegressor(
            #     hidden_units=[20],
            self._model = tf.estimator.LinearRegressor(
                feature_columns=self._feature_col,
                model_dir=PATH,
                optimizer=tf.train.AdamOptimizer(learning_rate=LEARNINGRATE)
                #warm_start_from=PATH
            )

        return self

    # define Train Spec
    def set_train_spec(self, train_spec):
        self._train_spec = train_spec
        
    # define Eval Spec
    def set_eval_spec(self, eval_spec):
        self._eval_spec = eval_spec

    # Train and Evaluate function
    def train_and_evaluate(self):
        if self._eval_spec == None or self._train_spec == None:
            print("Please instantiate eval and train specs.")
        else: 
            tf.estimator.train_and_evaluate(self._model, self._train_spec, self._eval_spec)