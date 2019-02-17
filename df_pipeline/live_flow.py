##############################################
# Dataflow pipeline to process both live 
# and historical data 
#      ^..^
# _||__(oo)____||___
# -||--"--"----||---
# _||_( __ )___||___
# -||--"--"----||---   
#  ||          ||
#
# Author: Even Tang 
# Feb 2018 
#############################################
# Merge tables 
# Delete columns
# Parse Date 
#############################################


import argparse
import logging
import datetime, os
import apache_beam as beam
import math
# import cloudstorage as gcs
import csv
from dateutil.parser import parse
from apache_beam.io.gcp.internal.clients import bigquery


##############################################
# Below are needed environment variables that 
# is defined
#############################################
PUBSUB_TOPIC = "projects/rich-principle-225813/topics/rossmann_real_time"
BUCKET_FILENAME = "store.csv"
CLOUD_BUCKET = 'gs://rossmann-cbd/store.csv'
STORE_INFO = []
PROJECT = "rich-principle-225813"
BUCKET = "live-data"
RUNNER = "DataFlowRunner"

# BigQuery Variables
BIGQUERY_LINK = 'live_test.test_live_rossman'
BIGQUERY_COLUMNS = [
    'CompetitionDistance', 
    'Year', 
    'Open"', 
    'Promo', 
    'SchoolHoliday', 
    'Promo2', 
    'Assortment', 
    'StateHoliday', 
    'StoreType', 
    'DayOfWeek', 
    'Month', 
    'WeekofYear', 
    'Sales'
]

# Filtering parameters
UNWANTED_COL_STORE = [0, 4, 5, 7, 8, 9]
# Wanted column order
COL_ODR = [8, 10, 2, 3, 5, 9, 7, 4, 6, 1, 11, 12, 0]
# Wanted types
TYPES = [1,1,1,1,1,1,0,0,0,1,1,1,1]

# Get commandline arguments for the pipeline 
def parseargs():
    return [
            '--project={0}'.format(PROJECT),
            '--job_name=live-rossmann',
           # '--save_main_session',
            '--staging_location=gs://{0}/run/'.format(BUCKET),
            '--temp_location=gs://{0}/temp/'.format(BUCKET),
            '--runner={0}'.format(RUNNER), 
            '--streaming'
        ]
    # Pardo transform and merge and return a list 
# !! Below functions assumes no missing values
class parse_live(beam.DoFn):
    def process(self, element):
        # Filtering and listing not needed columns
        data = [j[1:-1] if (i != 6 and i != 7) else j[2:-2] for i, j in enumerate(element.split(",")[1:])]
        store_id = int(data.pop(1)) - 1
        
        # Merging operation --> removes list
        data = data.append(STORE_INFO[store_id])
        return data

# Pardo transform on parsed date time columns 
class parse_date(beam.DoFn): 
    def process(self, element):
        date = parse(element.pop(2)) 
        return element.extend([date.year, date.month, date.isocalendar()[1]]) 
        '''
        Structure: [
            0 "Sales",
            1 ""DayOfWeek"",
            2 ""Open"",
            3 ""Promo"",
            4 ""StateHoliday"",
            5 ""SchoolHoliday"", 
            6 'StoreType', 
            7 'Assortment', 
            8 'CompetitionDistance', 
            9 'Promo2', 
            10 Year, 
            11 Month, 
            12 WeekofYear
        ]
        ''' 
# Pardo transform on parsed date time columns 
class corr_format(beam.DoFn): 
    def process(self, element):
        # Sorting to correct channel
        corr_col = [k for i in range(0, len(element)) for s,k in enumerate(element) if s == COL_ODR[i]]
        
        # Set correct types of columns 
        corr_col = [type_corr(i, TYPES[i], j) for i ,j in enumerate(corr_col)]

        return {BIGQUERY_COLUMNS[i]: j for i,j in enumerate(corr_col)}

# Correcting types basing on bq requirements
# !! Can use function pointers but oh welp :/ 
def type_corr(num_value, id, val): 
    if (num_value == 4): return str(val[1:-1]) 
    if (num_value == 7): return int(val[1:-1]) 
    
    if (id == 1):
        return int(val)
    else: 
        return str(val)

# Function to return the BigQuery schema 
def get_bqschema(): 
    table_schema = bigquery.TableSchema()

    for x in range(0, len(BIGQUERY_COLUMNS)): 
        source_field = bigquery.TableFieldSchema()
        source_field.name = BIGQUERY_COLUMNS[x]
        source_field.type = 'INTEGER' if (TYPES[x] == 1) else 'STRING'
        source_field.mode = 'NULLABLE'
        table_schema.fields.append(source_field)

    return table_schema

# Functrion to parse things correctly 
def parse_func(k): return [int(j) if (i != 1 and i != 2) else j[1:-1] for i,j in enumerate(k.split(",")[:-4])]

# Filter un-needed columns from historical data set
def filter_col(line): return [i for j, i in enumerate(line) if j not in UNWANTED_COL_STORE]

# Correct the type of data in historical data
def correct_type(line): return [int(j) if (i != 0 and i != 1) else j[1:-1] for i,j in enumerate(line)]

# Parse the historical data for combination 
def parset_hist(text):
    # Sets of strings --> lines 
    lines = text.split('\n')
    
    # Put into lists of lists
    lines = [line.split(",")[:-4] for line in lines]  
    
    # Cleansing not needed columns 
    lines = [filter_col(line) for line in lines]

    # Type correction 
    lines = [correct_type(line) for line in lines]

    return lines

# Start main here 
if __name__ == "__main__":
        with beam.Pipeline(argv=parseargs()) as p: 

            # File processing pipeline 
            store_info = (p 
            | 'Read historical data' >> beam.io.ReadAllFromText(CLOUD_BUCKET)
            | 'Text processing' >> beam.FlatMap(lambda text: parset_hist(text))
            ) 



            # Read live data from Pub/sub
            (p 
                | 'Read live data' >> beam.io.ReadStringsFromPubSub(topic=PUBSUB_TOPIC)
                | 'Parse input' >> beam.ParDo(parse_live())
                | 'Parse Date column' >> beam.ParDo(parse_date())
                | 'Formatting and type correction' >> beam.ParDo(corr_format())
                | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                            BIGQUERY_LINK, 
                            schema=get_bqschema(),
                            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
            
            
            p.run()
            logging.getLogger().setLevel(logging.INFO)