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
	
* CloudML:
    - turn model into a package such that it can be trained/deployed on cloudML
		* requires an estimator, train_spec and eval_spec (check docs)
	- consider distributed training?
	
Ideas to look at:

* BigTable to store data:
    - if the data ingest has a high throughput or we require real-time analysis with low latency, this could be a good option to store data in before passing it into cloudML