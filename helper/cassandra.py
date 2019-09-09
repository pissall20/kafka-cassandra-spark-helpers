import datetime
from datetime import timedelta
from datetime import timezone

import pandas as pd
from cassandra.cluster import Cluster


def utc_to_local(utc_dt):
    """
    Convert the utc timestamps to local timestamp
    params: utc_dt: pandas._libs.tslibs.timestamps.Timestamp in UTC timezone
    return: pandas._libs.tslibs.timestamps.Timestamp type in local timezone
    """
    return utc_dt.to_pydatetime().replace(tzinfo=timezone.utc).astimezone(tz=None)


def retriever_python_driver_cassandra(
    address,
    port,
    key_space,
    table_name,
    start_stamp,
    pred_steps=4,
    change_time_zone=True,
):
    """This function uses python driver for cassandra to make a cql selection query in the cassandra
    This returns pred_steps number of points in a pandas dataframe after start stamp
    Params: address: IP address of the cassandra cluster
            port: port of the cassandra cluster
            key_space: key_space where the table is stored
            tablename: table that we want to query
            start_stamp: start timestamp as datetime.datetime() object
            pred_steps: used to calculate end timestamp
            change_time_zone: if true then change the time to local timezone
    return: pandas dataframe with sorted queries. both the timestamps are exclusive
    """
    cluster = Cluster(address, port)
    session = cluster.connect(key_space)
    end_stamp = start_stamp + datetime.timedelta(seconds=(pred_steps + 1))
    rows = session.execute(
        f"SELECT * FROM {table_name} WHERE key<'{end_stamp}' and key>'{start_stamp}' ALLOW FILTERING"
    )
    df = pd.DataFrame(list(rows))
    if df.empty:
        col_names = ["key", "value"]
        df = pd.DataFrame(columns=col_names)
    if change_time_zone:
        temp = list(map(utc_to_local, df["key"]))
        df["key"] = temp
    df.sort_values(by=["key"], inplace=True)
    return df


def selector_python_driver_cassandra(
    address, port, key_space, table_name, start_stamp, end_stamp, change_time_zone=True
):
    """
    This function uses python driver for cassandra to make a cql selection query in the cassandra
    change_time_zone if set to true will convert the retrieved values to local timezone
    Params: address: IP address of the cassandra cluster
            port: port of the cassandra cluster
            key_space: key_space where the table is stored
            table_name: table that we want to query
            start_stamp: start timestamp as datetime.datetime() object
            end_stamp: end timestamp as datetime.datetime() object
            change_time_zone: if true then change the time to local timezone
    return: pandas dataframe with sorted queries. both the timestamps are exclusive
    """
    cluster = Cluster(address, port)
    session = cluster.connect(key_space)
    # start_stamp = datetime.datetime.strptime(start_stamp, '%Y-%m-%d %H:%M:%S')
    # end_stamp = datetime.datetime.strptime(end_stamp, '%Y-%m-%d %H:%M:%S')
    rows = session.execute(
        "SELECT * FROM {0} WHERE key<'{1}' and key>'{2}' ALLOW FILTERING".format(
            table_name, end_stamp, start_stamp
        )
    )
    df = pd.DataFrame(list(rows))
    if df.empty:
        col_names = ["key", "value"]
        df = pd.DataFrame(columns=col_names)
    if change_time_zone:
        temp = list(map(utc_to_local, df["key"]))
        df["key"] = temp
    df.sort_values(by=["key"], inplace=True)
    return df


def retrieve_max_stamp(address, port, key_space, table_name, change_time_zone=True):
    cluster = Cluster(address, port)
    session = cluster.connect(key_space)
    rows = session.execute("SELECT * FROM {0}".format(table_name))
    df = pd.DataFrame(list(rows))
    if df.empty:
        col_names = ["key", "value"]
        df = pd.DataFrame(columns=col_names)
    if change_time_zone:
        temp = list(map(utc_to_local, df["key"]))
        df["key"] = temp
    df.sort_values(by=["key"], inplace=True)
    return df.iloc[-1, :]["key"]


def writer_python_driver_cassandra(
    address, port, keyspace, table_name, start_stamp, pred_steps, predictions
):
    cluster = Cluster(address, port)
    session = cluster.connect(keyspace)
    for i in range(pred_steps):
        session.execute(
            f"INSERT INTO {table_name} (key, value) VALUES ('{start_stamp + timedelta(seconds=i)}', {predictions[i]})"
        )
    return
