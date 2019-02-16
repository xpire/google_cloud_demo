# Justin Or Feb 2019

from . import *
import json
import datetime
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

if __name__ == "__main__":
    # define predictionInput
    # look at https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#PredictionInput
    # PredictionInput = {
    #     "dataFormat": DataFormat.JSON,
    #     "inputPaths": [
    #         "gs://rossmann-cbd/predictionOutputs/request.json"
    #     ],
    #     "outputPath": "gs://rossmann-cbd/predictionOutputs/results.json",
    #     "region": "use-central1",
    #     "modelName": MODEL_NAME,
    # }
    # job = { 'jobId' : "predictionJob1", 'predictionInput'}

    base = datetime.date(2015, 8, 1)
    date_list = [base + datetime.timedelta(days=x) for x in range(0, 7)]
    # print(generate_input_request(date_list, range(1, 1115 + 1)))

    with open("request.json", "w+") as f:
        f.write(generate_input_request(date_list, range(1, 1115 + 1)))
    
