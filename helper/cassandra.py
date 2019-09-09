from datetime import timedelta

import pandas as pd
from cassandra.cluster import Cluster


class CassandraDriver(object):

    def __init__(self, ip_address, port, key_space, table_name):
        self.ip_address = ip_address
        self.port = port
        self.key_space = key_space
        self.table_name = table_name
        self.session = None

    def _connect_to_db(self):
        """
        Achieve a connected to the cassandra db, private method
        :return: A cassandra DB session object
        """
        cluster = Cluster(self.ip_address, self.port)
        session = cluster.connect(self.key_space)
        return session

    def connect_to_db(self):
        """
        Public interface (crude singleton) for database connection with Cassandra DB
        :return: A cassandra DB session object
        """
        if not self.session:
            self.session = self._connect_to_db()
        return self.session

    def retrieve(self, start_timestamp, end_timestamp, remove_tzinfo=True):
        """
        Make a cql selection query in the cassandra
        change_time_zone if set to true will convert the retrieved values to local timezone
        :param start_timestamp: start timestamp as datetime.datetime() object
        :param end_timestamp: end timestamp as datetime.datetime() object
        :param remove_tzinfo: if true then remove the timezone info
        :return: pd.DataFrame with sorted dates exclusive of both timestamps
        """
        session = self.connect_to_db()
        rows = session.execute(
            f"SELECT * FROM {self.table_name} WHERE key<'{end_timestamp}' and key>'{start_timestamp}' ALLOW FILTERING"
        )
        if not rows:
            raise ValueError("No rows were returned from the database")
        df = pd.DataFrame(list(rows))

        if remove_tzinfo:
            df["key"] = df["key"].replace(tzinfo=None)
        df.sort_values(by=["key"], inplace=True)
        return df

    def get_last_timestamp(self, remove_tzinfo=True):
        """
        Get the last timestamp existing in the database
        :param remove_tzinfo: Cassandra returns timestamps with UTC TZ. Do you wish to remove tz info?
        :return: max(date_index)
        """
        session = self.connect_to_db()
        rows = session.execute(f"SELECT * FROM {self.table_name}")
        if not rows:
            raise ValueError("No rows were returned from the database")
        df = pd.DataFrame(list(rows))

        if remove_tzinfo:
            df["key"] = df["key"].replace(tzinfo=None)
        return df["key"].max()

    def writer(self, start_timestamp, pred_steps, predictions):
        """
        Writes rows to Cassandra DB
        :param start_timestamp:
        :param pred_steps:
        :param predictions: Predictions from the forecasting engine
        :return: None
        """
        session = self.connect_to_db()
        for i in range(pred_steps):
            session.execute(
                f"INSERT INTO {self.table_name} (key, value) VALUES"
                f" ('{start_timestamp + timedelta(seconds=i)}', {predictions[i]})"
            )
