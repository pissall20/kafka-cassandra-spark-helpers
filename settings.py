# Settings

CASSANDRA_IP = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KEY_SPACE = "eugenie_iot"
CASSANDRA_TABLE_NAME = "demo_table_1"

INITIAL_TRAINING_POINTS = 1500
PREDICTION_STEPS_IN_SECONDS = 12

TIME_COLUMN = "key"

MODEL_LOCATION = "models/"
MODEL_ARCHIVE = "archive/"

TABLE_SCHEMA = {"id": "text", "key": "timestamp", "value": "double"}
