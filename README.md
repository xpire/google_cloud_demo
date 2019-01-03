##Cloud BD Solutions Demo Readme##

What is done:

* a linear regression model which predicts price based on store id and date
* wrap model in an API call for use in package

To be implemented:

* PUB/SUB: 
    - required for ingest of data (initially batch historical data stored in BigQuery or a bucket, and published to the one topic with timestamp as the one given within the data)
	
* Dataflow:
    - apache beam pipeline to pass data from Pub/Sub into the model, for training/evaluation
	- apply preprocessing to the data, such that when we move to serving, this pipeline can be used for **streaming** data
	
* BigQuery:
    - to store the data in, a good choice for large petabyte datasets
	
* CloudML:
    - turn model into a package such that it can be trained/deployed on cloudML
		* requires an estimator, train_spec and eval_spec (check docs)
	- consider distributed training?
	- note, remember the serving data skew, match the conditions of training as close to the serving scenario.
	
Ideas to look at:

* BigTable to store data:
    - if the data ingest has a high throughput or we require real-time analysis with low latency, this could be a good option to store data in before passing it into cloudML
	
	
##TIMELINE##

1. turn tf model into an estimator so that it can be put on cloudML
2. pub/sub program to publish to dataflow
3. dataflow pipeline that can:
    - read from pub/sub
	- do some preprocessing to the data
	- provide dashboard stats
	- push the data to model and bigquery for storage (if we were streaming, we would want to store the data that was processed)
4. consider cloudML distributed training by providing command line arguments for hyperparameters
5. consider datalab for dashboard