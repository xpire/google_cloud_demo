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
# import csv
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

# Get commandline arguments for the pipeline 
def parseargs():
    return [
            '--project={0}'.format(PROJECT),
            '--job_name=live-rossmann',
            '--save_main_session',
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
        return [i for i in enumerate(element.split(",")[1:])]

        # [Sales,Store,DayOfWeek,Date,Open,Promo,StateHoliday,SchoolHoliday]

# Pardo transform on parsed date time columns 
class parse_date(beam.DoFn): 
    def process(self, element):
        date = parse(element.pop(3)) 
        return element.extend([date.year, date.month, date.isocalendar()[1]])

        # [Sales,Store,DayOfWeek,Open,Promo,StateHoliday,SchoolHoliday,Years,Month,WeekOfYear]
        
# Pardo transform on parsed date time columns 
class corr_format(beam.DoFn): 
    def process(self, element):
        # correcting type
        corr_col = [int(k) if k != 6 else str(k) for k in enumerate(element)]
        
        return {'live_data': corr_col}

        # [Sales,Store,DayOfWeek,Open,Promo,s(StateHoliday),SchoolHoliday,Years,Month,WeekOfYear]

# Pardo on 2 different pcollections on transforms 
class merge_col(beam.DoFn): 
    def process(self, element): 
        
        live = None
        store = None  
        
        for k in element: 
            if 'live_data' in k: 
                live = k['live_data']
            elif 'store' in k:
                store = k['store']
        
        '''
        [
            0 s(StoreType),
            1 s(Assortment),
            2 CompetitionDistance,
            3 Promo2,
            4 Sales,
            5 DayOfWeek,
            6 Open,
            7 Promo,
            8 s(StateHoliday),
            9 SchoolHoliday,
            10 Year,
            11 Month,
            12 WeekOfYear
        ]

        '''

        merged = live_data.append(store[live.pop(1)])
        
        # Sorting to correct column
        corr_col = [k for i in range(0, len(element)) for s,k in enumerate(element) if s ==  [2, 10, 6, 7, 9, 3, 1, 8, 0, 5, 11, 12,4][i]]
        
        
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
        
        return {BIGQUERY_COLUMNS[i]: j for i,j in enumerate(corr_col)}

# Function to return the BigQuery schema 
def get_bqschema(): 

    # BigQuery Variables
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

    table_schema = bigquery.TableSchema()

    for x in range(0, len(BIGQUERY_COLUMNS)): 
        source_field = bigquery.TableFieldSchema()
        source_field.name = BIGQUERY_COLUMNS[x]
        source_field.type = 'INTEGER' if ([1,1,1,1,1,1,0,0,0,1,1,1,1][x] == 1) else 'STRING'
        source_field.mode = 'NULLABLE'
        table_schema.fields.append(source_field)

    return table_schema

# Functrion to parse things correctly 
def parse_func(k): return [int(j) if (i != 1 and i != 2) else j[1:-1] for i,j in enumerate(k.split(",")[:-4])]

# Filter un-needed columns from historical data set
def filter_col(line): return [i for j, i in enumerate(line) if j not in [0, 4, 5, 7, 8, 9]]

# Correct the type of data in historical data
def correct_type(line): return [int(float(j)) if (i != 0 and i != 1) else j for i,j in enumerate(line)]

# Parse the historical data for combination 
class parset_hist(beam.DoFn): 
    def process(self, element): 
        # !! Get rid of header 
        lines = element.split('\n')[1:]
        
        # Put into lists of lists
        # [Store,StoreType,Assortment,CompetitionDistance,CompetitionOpenSinceMonth,CompetitionOpenSinceYear,Promo2,Promo2SinceWeek,Promo2SinceYear]
        lines = [line.split(",")[:-4] for line in lines]  
        
        # Cleansing not needed columns 
        # [StoreType,Assortment,CompetitionDistance,Promo2]
        lines = [filter_col(line) for line in lines]

        # Type correction 
        # [s(StoreType),s(Assortment),CompetitionDistance,Promo2]
        lines = [correct_type(line) for line in lines]

        return {'store': lines}

# Start main here 
if __name__ == "__main__":
        with beam.Pipeline(argv=parseargs()) as p: 

            BIGQUERY_LINK = 'live_test.test_live_rossman'

            live_data = (p
                | "Read live data" >> beam.io.ReadStringsFromPubSub(topic=PUBSUB_TOPIC)
                | 'Parse input' >> beam.ParDo(parse_live()) 
                | 'Parse Date column' >> beam.ParDo(parse_date())
                | 'Formatting and type correction' >> beam.ParDo(corr_format())
            )
            
            # File processing pipeline 
            store_info = (p 
                | 'Read historical data' >> beam.io.ReadAllFromText(CLOUD_BUCKET)
                | 'Text processing' >> beam.ParDo(parset_hist())
            ) 
        
            # Read live data from Pub/sub
            (
                (live_data, store_info)
                | "Flattern data" >> beam.Flatten()
                | 'Merge column' >> beam.ParDo(merge_col())
                | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                            BIGQUERY_LINK, 
                            schema=get_bqschema(),
                            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
            
            p.run()
            logging.getLogger().setLevel(logging.INFO)