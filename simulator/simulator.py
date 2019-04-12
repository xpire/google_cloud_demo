############################################################
# A simple sctipt for live data publishing onto Pub/Sub
############################################################
from google.cloud import pubsub_v1
import csv
import time

PROJECT_ID = "big-data-demo-219402"
TOPIC_NAME = "real_time_publish"
FILE_NAME = "live_set.csv"


if __name__ == "__main__":

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

    # Open file
    live_file = open(FILE_NAME)
    line_count = 0

    # Sending in each line of the file
    for line in live_file:
        if (line_count != 0): 

            # Data must be a bytestring
            data = line.encode('utf-8')

            # When you publish a message, the client returns a future.
            future = publisher.publish(topic_path, data=data)
            print('Published {}.'.format(data))

            time.sleep(60)

        line_count += 1

    print("All data is done.")