# Justin Or Feb 2019

from . import *
import json
import datetime
import glob
from google.cloud import storage
import re

# Given rows of information of the form: 
    # Date (Year, DayOfWeek, Month, WeekOfYear)
    # StoreID (CompetitionDistance, Promo2, Assortment, Storetype)
    # Open
    # Promo2
    # StateHoliday
    # SchoolHoliday
# Goal: I want to create functions to
    # create the inputs to the model for some future dates for specified data
    # a function to run the requests through the model and save the results in gcs
    # somehow get this data to show up on data studio

# Usage: gcloud ml-engine jobs submit prediction PREDICTION3 --model rossmann_cbd_test_7 --input-paths gs://rossmann-cbd/predictionOutputs/request.json --output-path gs://rossmann-cbd/predictionOutputs/results --region us-central1 --data-format text
# Readable version:
# gcloud ml-engine jobs submit prediction PREDICTION1 \
# --model rossmann_cbd_test_7 \
# --input-paths gs://rossmann-cbd/predictionOutputs/request.json \
# --output-path gs://rossmann-cbd/predictionOutputs/results \
# --region us-central1 \
# --data-format text

# batch job prediction
PROJECT_NAME = "projects/rich-principle-225813"
MODEL_NAME = PROJECT_NAME + "/models/rossmann_cbd_test_7"
VERSION_NAME = MODEL_NAME + "/versions/rossmann_cbd_test_7"
BUCKET_NAME  = "gs://rossmann-cbd/"
SOURCE_FILE = "request.json"
DESTINATION_BLOB = "predictionOutputs/request.json"
GOOGLE_APPLICATION_CREDENTIALS = 
# SET THE GOOGLE_APPLICATION_CREDENTIALS: https://cloud.google.com/docs/authentication/getting-started

# Combine dictionaries of the request and output as a string
def generate_input_request(date_list, storeID_list):
    i = 0
    final_result = ""
    for date in date_list:
        for storeID in storeID_list:
            date_res = parseTime(date)
            store_res = store[storeID]
            infer_res = infer_data(storeID, date)
            results = {**date_res, **store_res, **infer_res}
            # results["key"] = i
            i += 1
            output = json.dumps(results)
            final_result += output + "\n"
    return final_result

# From https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py
# https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
# [START storage_upload_file]
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(credentials=GOOGLE_APPLICATION_CREDENTIALS)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
# [END storage_upload_file]


if __name__ == "__main__":
    # define predictionInput
    # from https://cloud.google.com/ml-engine/docs/tensorflow/batch-predict
    # look at https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#PredictionInput
    # Configuring job data
    PredictionInput = {
        "dataFormat": "JSON",
        "inputPaths": [
            "gs://rossmann-cbd/predictionOutputs/request.json"
        ],
        "outputPath": "gs://rossmann-cbd/predictionOutputs/results",
        "region": "us-central1",
        "modelName": MODEL_NAME,
    }

    # Generate the list of dates to predict for
    base = datetime.date(2015, 8, 1)
    date_list = [base + datetime.timedelta(days=x) for x in range(0, 7)]
    # print(generate_input_request(date_list, range(1, 1115 + 1)))

    # Generate the list of StoreIds to predict for
    store_list = range(1, 1115 + 1)

    # Write to a json file
    # with open("request.json", "w+") as f:
    #     f.write(generate_input_request(date_list, store_list))

    # upload file to gcs folder
    upload_blob(BUCKET_NAME, SOURCE_FILE, DESTINATION_BLOB)

    # Job name is in the form: "<model_name>_batch_pred_YYYYMMDD_HHMMSS"
    job_name = re.search("[a-z\-]+$", MODEL_NAME)
    print(job_name)
    job = { 'jobId' : "predictionJob1", 'predictionInput' : PredictionInput}

    # RUN JOB ON CLOUD ML


    