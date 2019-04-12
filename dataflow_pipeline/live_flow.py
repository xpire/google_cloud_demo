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

import argparse
import logging
import datetime, os
import apache_beam as beam
from apache_beam import pvalue
import math
# import cloudstorage as gcs
# import csv
# from dateutil.parser import parse
from apache_beam.io.gcp.internal.clients import bigquery


##############################################
# Below are needed environment variables that 
# is defined
#############################################
STORE_INFO = []
PROJECT = "big-data-demo-219402"
BUCKET = "demo-dataflow-data"
RUNNER = "DataFlowRunner"



# Get commandline arguments for the pipeline 
def parseargs():
    return [
            '--project={0}'.format(PROJECT),
            '--job_name=live-publish',
            #'--save_main_session',
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
        # [Sales,Store,DayOfWeek,Date,Open,Promo,StateHoliday,SchoolHoliday,Date]
        corr_col = [i for i in element.split(",")[1:]]

        # Parse Date
        # !! Date time might become a problem here
        # [Sales,Store,DayOfWeek,Open,Promo,StateHoliday,SchoolHoliday,Years,Month,WeekOfYear,Date]
        from dateutil.parser import parse

        res_date = corr_col.pop(3)
        date = parse(res_date) 
        corr_col.append(int(date.year))
        corr_col.append(int(date.month))
        corr_col.append(int(date.isocalendar()[1]))
        corr_col.append(res_date)


        # correcting type
        # [Sales,Store,DayOfWeek,Open,Promo,s(StateHoliday),SchoolHoliday,Years,Month,WeekOfYear,s(Date)]
        corr_col = [str(j) if (k == 5  or k == 10) else int(j) for k,j in enumerate(corr_col)]

        yield corr_col



# Pardo on 2 different pcollections on transforms 
class merge_col(beam.DoFn): 
    def process(self, element): 
        # Filter un-needed columns from historical data set
        def filter_col(line): return [i for j, i in enumerate(line) if j not in [4, 5, 7, 8, 9]]

        # Correct the type of data in historical data
        def correct_type(line): return [int(float(j)) if (i != 1 and i != 2) else j for i,j in enumerate(line)]

        # Parse the historical data for combination 
        def parset_hist(line): 
                
            # Put into lists of lists
            # [Store,StoreType,Assortment,CompetitionDistance,CompetitionOpenSinceMonth,CompetitionOpenSinceYear,Promo2,Promo2SinceWeek,Promo2SinceYear]
            line = line.split(",")
            if (len(line) == 13):
                line = line[:-4] 
            

            # Cleansing not needed columns 
            # [Store, StoreType,Assortment,CompetitionDistance,Promo2]
            line = filter_col(line) 

            # Type correction 
            # [Store,s(StoreType),s(Assortment),CompetitionDistance,Promo2]
            line = correct_type(line) 
            
            return line        

        CLOUD_BUCKET = 'gs://demo-dataflow-data/store.csv'
        import apache_beam as beam

        store = []

        with beam.io.gcp.gcsio.GcsIO().open(CLOUD_BUCKET, 'r') as f:
            
            line_count = 0
            for line in f: 
                if (line_count == 0):
                    line_count += 1
                else:
                    store.append(str(line))
           
                print(line +"\n") 

        store_id = element.pop(1) - 1
        for k in parset_hist(store[store_id]): 
            element.append(k)

        '''
        ['Sales'0, 'DayOfWeek'1, 'Open'2,'Promo'3,'s(StateHoliday)'4,'SchoolHoliday'5,'Year'6,'Month'7,'WeekOfYear'8,Date 9,'Store'10,'s(StoreType)'11,'s(Assortment)'12,'CompetitionDistance'13,'Promo2'14]
        '''
        # Sorting to correct column
        corr_col = [k for i in range(0, len(element)) for s,k in enumerate(element) if s == [13, 6, 2, 3, 5, 14, 12, 4, 11, 1, 7, 8, 0, 10, 9][i]]
        
        BIGQUERY_COLUMNS = [
            'CompetitionDistance', 
            'Year', 
            'Open', 
            'Promo', 
            'SchoolHoliday', 
            'Promo2', 
            'Assortment', 
            'StateHoliday', 
            'StoreType', 
            'DayOfWeek', 
            'Month', 
            'WeekOfYear', 
            'Sales',
            'Store',
            'Date'
        ]

        yield {BIGQUERY_COLUMNS[i]: j for i,j in enumerate(corr_col)}



# Filter 2 columns again 
class Filter_2(beam.DoFn): 
    def process(self, element):
        del element['Store'] 
        del element['Date']
        yield element


# Function to return the BigQuery schema; 0 for training and test set, 1 for prediction and visualization  
def get_bqschema(flag): 

    # Type detection 
    def type_detect(type_def, column_id):
        if (type_def[column_id] == 1): return 'INTEGER'
        elif (type_def[column_id] == 0 ): return 'STRING'
        else: return 'DATE'
        
    type_def = [1,1,1,1,1,1,0,0,0,1,1,1,1,1,-1]
    # BigQuery Variables
    BIGQUERY_COLUMNS = [
        'CompetitionDistance', 
        'Year', 
        'Open', 
        'Promo', 
        'SchoolHoliday', 
        'Promo2', 
        'Assortment', 
        'StateHoliday', 
        'StoreType', 
        'DayOfWeek', 
        'Month', 
        'WeekOfYear', 
        'Sales',
        'Store',
        'Date'
    ]

    if (flag == 0) :  
        type_def = type_def[:-2]
        BIGQUERY_COLUMNS = BIGQUERY_COLUMNS[:-2]

    
    table_schema = bigquery.TableSchema()
    for x in range(0, len(BIGQUERY_COLUMNS)): 
        source_field = bigquery.TableFieldSchema()
        source_field.name = BIGQUERY_COLUMNS[x]
        source_field.type = type_detect(type_def, x)
        source_field.mode = 'NULLABLE'
        table_schema.fields.append(source_field)

    return table_schema



class Split(beam.DoFn):
    # These tags will be used to tag the outputs of this DoFn.
    OUTPUT_TAG_TRAINING = 'TrainingSet'
    OUTPUT_TAG_TESTING = 'TestSet'

    def process(self, element):
        """
        tags the input as it processes the orignal PCollection
        """
        import random 
    
        split_id = random.randint(1,101)

        # Conditional for testing and training test split 
        if split_id <= 80:
            yield pvalue.TaggedOutput(self.OUTPUT_TAG_TRAINING, element)

        else:
            yield pvalue.TaggedOutput(self.OUTPUT_TAG_TESTING, element)



####################################################
# Outputting to 4 different tables  
# Training table 
# Testing table
# Aggregate table  
# Prediction table
####################################################
if __name__ == "__main__":
        with beam.Pipeline(argv=parseargs()) as p: 
            # visualization of both historical data and new data
            BIGQUERY_LINK_AGGREGATE = 'demo_data_US.TotalDisplay'
            BIGQUERY_LINK_TRAINING = 'demo_data_US.TrainingSet'
            BIGQUERY_LINK_TESTING = 'demo_data_US.TestingSet'
            BIGQUERY_LINK_PREDICTION = 'demo_data_US.PredictionDisplay'

            PUBSUB_TOPIC_LIVE = "projects/big-data-demo-219402/topics/real_time_publish"
            PUBSUB_TOPIC_PRED = '' # TODO 
            
            # default set indicators 
            TRAIN_TEST = 0
            PRED_AGGRE = 1
            

            # Training and Testing table parsing, write to aggregate table
            live_data = (p
                | "Read live data" >> beam.io.ReadStringsFromPubSub(topic=PUBSUB_TOPIC_LIVE)
                | 'Parse input' >> beam.ParDo(parse_live()) 
                | 'Merge column' >> beam.ParDo(merge_col())
            )

            # Probability to write conditionally training or testing
            prob_write = (live_data
                | 'Filter out the last columns' >> beam.ParDo(Filter_2())
                | 'Randomly split' >> beam.ParDo(Split()).with_outputs(
                        Split.OUTPUT_TAG_TRAINING,
                        Split.OUTPUT_TAG_TESTING
                )
            )

            write_to_train = prob_write[Split.OUTPUT_TAG_TRAINING] | "Write to training dataset" >> beam.io.WriteToBigQuery(
                    BIGQUERY_LINK_TRAINING, 
                    schema=get_bqschema(TRAIN_TEST),
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            
            write_to_test = prob_write[Split.OUTPUT_TAG_TESTING] | "write Testing dataset" >> beam.io.WriteToBigQuery(
                    BIGQUERY_LINK_TESTING, 
                    schema=get_bqschema(TRAIN_TEST),
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )

            # Write to aggregate table  
            write_to_aggregate = (live_data 
                | 'Write to the aggregate BigQuery table' >> beam.io.WriteToBigQuery(
                    BIGQUERY_LINK_AGGREGATE, 
                    schema=get_bqschema(PRED_AGGRE),
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )

            # Prediction phrase, this needs the original columns
            prediction_parse = (p   
                | "Read from prediction topic" >> beam.io.ReadStringsFromPubSub(topic=PUBSUB_TOPIC_PRED)
                | 'Parse input' >> beam.ParDo(parse_live()) 
                | 'Merge column' >> beam.ParDo(merge_col()) 
                | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                    BIGQUERY_LINK_PREDICTION, 
                    schema=get_bqschema(PRED_AGGRE),
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                     write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                )
            )
    
            logging.getLogger().setLevel(logging.INFO)