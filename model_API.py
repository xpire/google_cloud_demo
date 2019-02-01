"""
Simple wrapper API for choosing different models 
(maybe) improve for later escaping))
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

PATH = "./model_check/"

"""
Importing libraries
"""
import tensorflow as tf
import numpy as np
import pandas as pd

# Model for fixed model
class model:
    def __init__(self):
        # currently what type of optimizer ?
        self._feature_col = None 
        self._model = None

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
            self._model = tf.estimator.LinearRegressor(
                feature_columns=self._feature_col,
                model_dir=PATH
            )

        return self

    # Training model
    def train(self, input_fn=None): 
        # Input function
        if (input_fn == None or self._model == None): 
            print("No input function.")
        else:
            self._model.train(input_fn=input_fn)


    # evaluation
    def eval(self, input_fn=None):
        if (self._model == None or input_fn == None):
            print("Please set model or redefine input function. ")
        else:
            result = self._model.evaluate(input_fn=input_fn)

            # print("Evaluation results")
            # for key in result:
            #     print("   {}, was: {}".format(key, result[key]))

    # Prediction using the model
    def predict(self, input_fn=None):
        if (self._model == None or input_fn == None):
            print("Please set model or redefine input function. ")
        else:
            self._model.predict(input_fn=input_fn)