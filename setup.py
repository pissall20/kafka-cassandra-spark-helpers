import pandas as pd
import numpy as np

import settings
from helper.cassandra import CassandraInterface

new_table_schema = settings.TABLE_SCHEMA


def create_random_data():
    p_df = pd.DataFrame()
    id_suffix = "device_"

    start_time, end_time = pd.to_datetime("2018-12-28"), pd.to_datetime("2018-12-31")
    date_range = pd.date_range(start_time, end_time, freq="S")
    rows_per_id = len(date_range)

    ids = range(1, 11)
    for s_id in ids:
        item_id = id_suffix + str(s_id)
        values = np.random.random(size=(rows_per_id, 1)) * 100
        temp_df = pd.DataFrame(columns=new_table_schema.keys())
        temp_df["id"] = [item_id] * rows_per_id
        temp_df["key"] = date_range
        temp_df["value"] = values
        p_df = p_df.append(temp_df)
    return p_df.reset_index(drop=True)


if __name__ == "__main__":
    cql_connect = CassandraInterface(settings.CASSANDRA_IP, settings.CASSANDRA_PORT)
    new_keyspace, new_table = settings.CASSANDRA_KEY_SPACE, settings.CASSANDRA_TABLE_NAME
    cql_connect._create_key_space(new_keyspace)
    cql_connect.key_space = new_keyspace

    cql_connect._create_table(
        new_table,
        schema=new_table_schema,
        primary_key_cols=[col for col in new_table_schema.keys() if col != "value"],
    )

    df = create_random_data()
    col_names = ", ".join(df.columns)
    query = f"INSERT INTO {new_table}({col_names}) VALUES ({', '.join(['?'] * len(df.columns))});"
    prepared_query = cql_connect.session.prepare(query)

    for row in df.iterrows():
        row = row[1]
        cql_connect.session.execute(prepared_query, row.values.tolist())
