##############################################
# Dataflow pipeline to process both live 
# and historical data 
#      ^..^
# _||__(oo)____||___
# -||--"--"----||---
# _||_( __ )___||___
# -||--"--"----||---   
#  ||          ||
#############################################

import argparse
import logging
import datetime, os
import apache_beam as beam
import math

if __name__ == "__main__":
    