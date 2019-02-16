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
import assembly as asb
import cloudstorage as gcs
import csv

from dateutil.parser import parse
from apache_beam.io.gcp.internal.clients import bigquery




##############################################
# Below are needed environment variables that 
# is defined
#############################################
PUBSUB_TOPIC = "rossmann_real_time"
BUCKET_FILENAME = "/rossmann-cbd/store.csv"

STORE_INFO = []

PROJECT = "rich-principle-225813"
BUCKET = "live-data"
RUNNER = "DataFlowRunner"

# BigQuery Variables
BIGQUERY_LINK = 'rich-principle-225813.live_test.test_live_rossman'
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
            '--job_name=live_rossmann',
            '--save_main_session',
            '--staging_location=gs://{0}/run/'.format(BUCKET),
            '--temp_location=gs://{0}/temp/'.format(BUCKET),
            '--runner={0}'.format(RUNNER)
        ]

# Pardo transform and merge and return a list 
# !! Below functions assumes no missing values
class parse_live(beam.DoFn):
    def process(self, element):
        # Filtering and listing not needed columns
        data = list(csv.reader([element]))[0][1:]
        store_id = data.pop(1) - 1
        
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

# Start main here 
if __name__ == "__main__":

    #!! Read from cloud storage check if csv files are compatible  
    with gcs.open(BUCKET_FILENAME) as store_info:
        content = csv.reader(store_info, delimiter=',')
        line_count = 0 
        
        # Process each row and filters each column
        for row in content: 
            if (line_count != 0): 
                STORE_INFO.append([i for j, i in enumerate(row) if j not in UNWANTED_COL_STORE])
                

    with beam.Pipeline(argv=asb.parseargs()) as p: 
    
        # Read live data from Pub/sub
        plive = (p 
                    | 'Read live data' >> beam.io.ReadFromPubSub(topic=PUBSUB_TOPIC)
                    | 'Parse input' >> beam.ParDo(parse_live())
                    | 'Parse Date column' >> beam.ParDo(parse_date())
                    | 'Formatting and type correction' >> beam.Pardo(corr_format())
                    | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                        BIGQUERY_LINK, 
                        schema=get_bqschema(),
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                    )
                )
        
        
        #p.run()
        #logging.getLogger().setLevel(logging.INFO)
