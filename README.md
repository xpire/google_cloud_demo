# Cloud BD Solutions Demo Readme

What is done:

* a linear regression model which predicts price based on store id and date

* wrap model in an API call for use in package

* PUB/SUB:
    - required for ingest of data (initially batch historical data stored in BigQuery or a bucket, and published to the one topic with timestamp as the one given within the data)

* Dataflow:
    - apache beam pipeline to pass data from Pub/Sub into the model, for training/evaluation
        - apply preprocessing to the data, such that when we move to serving, this pipeline can be used for **streaming** data

* BigQuery to store the data in

* CloudML:
    - Model written as a package to be trained/deployed on cloudML (including estimator, train_spec and eval_spec (check docs))
        - note, remember the serving data skew, match the conditions of training as close to the serving scenario.

* Datalab for visualisations

* Datastudio for dashboard

## Ideas to look at:

* BigTable to store data:
    - if the data ingest has a high throughput or we require real-time analysis with low latency, this could be a good option to store data in before passing it into cloudML
