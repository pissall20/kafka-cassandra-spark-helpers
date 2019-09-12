import pandas as pd
import numpy as np

import settings
from helper.cassandra import CassandraInterface

new_table_schema = settings.TABLE_SCHEMA


def create_random_data():
    p_df = pd.DataFrame()
    main_id_suffix = "device_"
    second_id_suffix = "sensor_"

    # Define a date range
    start_time, end_time = (
        pd.to_datetime("2018-12-31 00:00:00"),
        pd.to_datetime("2018-12-31 00:10:00"),
    )
    date_range = pd.date_range(start_time, end_time, freq="S")
    rows_per_id = len(date_range)
    print(f"Number of rows per unique combination: {rows_per_id}")

    # 10 unique ID's
    ids = range(1, 11)
    for p_id in ids:
        main_id = main_id_suffix + str(p_id)
        for s_id in ids:
            item_id = second_id_suffix + str(s_id)
            values = np.random.random(size=(rows_per_id, 1)) * 100
            temp_df = pd.DataFrame(columns=new_table_schema.keys())
            temp_df["primary_id"] = [main_id] * rows_per_id
            temp_df["secondary_id"] = [item_id] * rows_per_id
            temp_df["key"] = date_range
            temp_df["value"] = values
            p_df = p_df.append(temp_df)
    return p_df.reset_index(drop=True)


if __name__ == "__main__":
    cql_connect = CassandraInterface(settings.CASSANDRA_IP, settings.CASSANDRA_PORT)
    new_keyspace, new_table = (
        settings.CASSANDRA_KEY_SPACE,
        settings.CASSANDRA_TABLE_NAME,
    )
    cql_connect._create_key_space(new_keyspace)
    cql_connect.key_space = new_keyspace

    # Create real values table
    cql_connect._create_table(
        new_table,
        schema=new_table_schema,
        primary_key_cols=[col for col in new_table_schema.keys() if col != "value"],
    )
    # Create table to store model information, as to which model was in use at what time
    cql_connect._create_table(
        "model_history",
        settings.MODEL_HISTORY_SCHEMA,
        primary_key_cols=settings.IDENTIFIER_GROUP,
    )

    df = create_random_data()
    cql_connect.write_rows(df, settings.CASSANDRA_TABLE_NAME, settings.TABLE_SCHEMA)
    print(f"Inserting {len(df)} rows")
