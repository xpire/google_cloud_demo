# Justin Or Feb 2019
# in datalab, inside big-data-demo CBD
# use this command to start it:
# datalab connect --zone us-central1-a --port 8081 prediction?

# TODO: host this script on google cloud functions
# https://cloud.google.com/functions/features/

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

from . import *
import json
import datetime
import glob
from google.cloud import storage
from google.cloud import pubsub_v1
import googleapiclient.discovery as discovery
import google.cloud.storage.client as gcs
import tensorflow as tf # Use tf's csv glob open method
import csv
import time
import re

# Combine dictionaries of the request and output as a string
def generate_input_request(date_list, storeID_list):
    i = 0
    final_result = ""
    set_result = []
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
            res_set = [i, storeID, date.weekday(), date, infer_res["Open"], infer_res["Promo"], infer_res["StateHoliday"], infer_res["SchoolHoliday"]]
            set_result.append(res_set)
    return final_result, set_result

# From https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py
# https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
# TODO: move to cloudstorage package if we stage this on appengine/google cloud functions:
#   https://cloud.google.com/appengine/docs/standard/python/googlecloudstorageclient/read-write-to-cloud-storage 
# [START storage_upload_file]
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
# [END storage_upload_file]

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.
    This can be used to list all blobs in a "folder", e.g. "public/".
    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:
        /a/1.txt
        /a/b/2.txt
    If you just specify prefix = '/a', you'll get back:
        /a/1.txt
        /a/b/2.txt
    However, if you specify prefix='/a' and delimiter='/', you'll get back:
        /a/1.txt
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    results = []

    # print('Blobs:')
    for blob in blobs:
        # print(blob.name)
        results.append(blob.name)

    if delimiter:
        # print('Prefixes:')
        for prefix in blobs.prefixes:
            # print(prefix)
            results.append(prefix)
    
    return results


def execute_request(request):
    try:
        response = request.execute()

        print('Job requested.')

        # The state returned will almost always be QUEUED.
        print('state : {}'.format(response['state']))

    except errors.HttpError as err:
        # Something went wrong, print out some information.
        print('There was an error getting the prediction results.' +
              'Check the details:')
        print(err._get_reason())

    return response

def find_max_shards(results_list):
    number = None
    for results_file in results_list:
        if OUTPUT_ID + "-" not in results_file:
            continue
        # this is a prediction file
        # find the total number of files
        print(results_file)
        end = re.search(r'[0-9]+$',results_file).group(0)
        number = int(end)
        print("number of shards = " + str(number))
        break
    if number == None:
        print("could not find the max number of shards.")
        exit(1)
    return number

if __name__ == "__main__":
    # define predictionInput
    # from https://cloud.google.com/ml-engine/docs/tensorflow/batch-predict
    # look at https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#PredictionInput
    # Configuring job data
    PredictionInput = {
        "dataFormat": "JSON",
        "inputPaths": [
            INPUT_PATH
        ],
        "outputPath": OUTPUT_PATH,
        "region": "us-central1",
        "modelName": MODEL_NAME,
    }

    # Generate the list of dates to predict for
    base = datetime.date(2015, 8, 1)
    date_list = [base + datetime.timedelta(days=x) for x in range(0, 7)]

    # Generate the list of StoreIds to predict for
    store_list = range(1, 1115 + 1)

    # Write to a json file
    source, live = generate_input_request(date_list, store_list)
    print(str(live[:5]) + "..." + str(live[-5:]))
    
    # prevent New lines being added after every row: https://stackoverflow.com/questions/30929363/csv-writerows-puts-newline-after-each-row
    with open(SOURCE_FILE, "w+") as f, open(LIVE_FILE, "w+", newline='') as l:
        f.write(source)
        csv_writer = csv.writer(l, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in live:
            csv_writer.writerow(line)

    # upload file to gcs folder
    upload_blob(BUCKET_NAME, SOURCE_FILE, DESTINATION_BLOB)

    # Job name is in the form: "<model_name>_batch_pred_YYYYMMDD_HHMMSS"
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    JOB_ID = MODEL_ID + "_batch_pred_" + timestamp
    JOB_BODY = { 'jobId' : JOB_ID, 'predictionInput' : PredictionInput}
    JOB_NAME = '{}/jobs/{}'.format(PROJECT_NAME, JOB_ID)

    # RUN JOB ON CLOUD ML (https://cloud.google.com/ml-engine/docs/tensorflow/batch-predict)
    ml = discovery.build('ml', 'v1')
    create_request = ml.projects().jobs().create(parent=PROJECT_NAME,
                                          body=JOB_BODY)

    response = execute_request(create_request)
    
    print("Job status for {}.{}".format(PROJECT_NAME, JOB_ID))
    start = time.time()
    while True:
        now = time.time()
        header = ""
        
        # Monitor Cloud ML job: (https://cloud.google.com/ml-engine/docs/tensorflow/monitor-training)
        get_request = ml.projects().jobs().get(name=JOB_NAME)
        response = execute_request(get_request)
        
        # enum of possible states: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#State
        payload = response['state']
        exit_cond = ""
        
        if response == None:
            header = "no response: "
            exit_cond = "exit"
        elif payload in ["QUEUED", "PREPARING", "RUNNING"]:
            header = "state: "
        elif payload == "SUCCEEDED":
            header = "state: "
            exit_cond = "break"
        else:
            # payload in ['FAILED', 'CANCELLED']:
            header = "failed state: "
            exit_cond = "exit"
        total = str(datetime.timedelta(seconds=(now - start)))
        print("Time: " + total + ", " + header + payload)
        if exit_cond == "exit":
            exit()
        elif exit_cond == "break":
            break
        time.sleep(5)
    
    # job succeeded
    # append results into one consistent file
    # https://medium.com/google-cloud/how-to-write-to-a-single-shard-on-google-cloud-storage-efficiently-using-cloud-dataflow-and-cloud-3aeef1732325
    files = list_blobs_with_prefix(BUCKET_NAME, prefix=OUTPUT_DIR)
    num_shards = find_max_shards(files)
    client = gcs.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = []
    for shard in range(num_shards):
        sfile = '%s-%05d-of-%05d' % (OUTPUT_DIR + "/" + OUTPUT_ID, shard, num_shards)
        blob = bucket.blob(sfile)
        if not blob.exists():
            # this causes a retry in 60s
            raise ValueError('Shard {} not present'.format(sfile))
        blobs.append(blob)
    bucket.blob(OUTPUT_NAME).compose(blobs)

    # List the new file
    files = list_blobs_with_prefix(BUCKET_NAME, prefix=OUTPUT_NAME)
    print(files)

    # Download this file
    download_blob(BUCKET_NAME, OUTPUT_NAME, OUTPUT_ID)

    # configuring Pub/Sub for batch (https://cloud.google.com/pubsub/docs/publisher#batching-to-balance-latency-and-throughput)
    batch_settings = pubsub_v1.types.BatchSettings(
        max_bytes=1024,  # One kilobyte
        max_latency=1,  # One second
    )

    # Setting up Pub/Sub
    publisher = pubsub_v1.PublisherClient(batch_settings=batch_settings)
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

    # Join results with requests and upload to Pub/Sub
    with open(OUTPUT_ID, 'r') as open_results, open(LIVE_FILE, 'r') as open_live:
        results_reader = csv.reader(open_results, delimiter=',', quotechar='|')
        live_reader = csv.reader(open_live, delimiter=',', quotechar='|')
        
        for source in live_reader:
            results = next(results_reader)
            print("source: "+ str(source))
            print("results: "+ str(results[0])) 
            results = json.loads(results[0])
            source_results = [source[0], round(results["predictions"][0])] + source[1:]
            comma_separated = ",".join(map(str, source_results))
            print(comma_separated)
            data = comma_separated.encode('utf-8')
            future = publisher.publish(topic_path, data=data, origin="prediction")

    print("Data has finished publishing.")

    # TODO: Data Studio doing custom queries (ie join predict with historical to get projections)
    # https://cloud.google.com/bigquery/docs/visualize-data-studio#create_a_chart_using_a_custom_query

    # TODO: push data from bigquery to csv (if larger than 1gb, needs to be separated into multiple files?)
    # https://cloud.google.com/bigquery/docs/exporting-data

