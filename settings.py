# Settings

CASSANDRA_IP = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KEY_SPACE = "eugenie_iot"
CASSANDRA_TABLE_NAME = "demo_table"

INITIAL_TRAINING_POINTS = 1500
PREDICTION_STEPS_IN_SECONDS = 12

MODEL_LOCATION = "models/"
MODEL_ARCHIVE = "archive/"

TABLE_SCHEMA = {
    "primary_id": "text",
    "secondary_id": "text",
    "key": "timestamp",
    "value": "double",
}

TIME_COLUMN = "key"
VALUE_COLUMN = "value"
IDENTIFIER_GROUP = ["primary_id", "secondary_id"]
