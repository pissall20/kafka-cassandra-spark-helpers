import os

# Settings

CASSANDRA_IP = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KEY_SPACE = "eugenie_iot"
CASSANDRA_TABLE_NAME = "demo_table"

INITIAL_TRAINING_POINTS = 1500
PREDICTION_STEPS_IN_SECONDS = 12

MODEL_LOCATION = "models/"
MODEL_ARCHIVE = os.path.join(MODEL_LOCATION, "archive")

TABLE_SCHEMA = {
    "primary_id": "text",
    "secondary_id": "text",
    "key": "timestamp",
    "value": "double",
}

TIME_COLUMN = "key"
VALUE_COLUMN = "value"
IDENTIFIER_GROUP = ["primary_id", "secondary_id"]

MODEL_HISTORY_SCHEMA = {
    **{"model_name": "text", "model_file": "text", "time": "timestamp"},
    **{
        col: col_type
        for col, col_type in TABLE_SCHEMA.items()
        if col in IDENTIFIER_GROUP
    },
}
