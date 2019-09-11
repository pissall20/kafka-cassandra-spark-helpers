import pandas as pd
from cassandra.cluster import Cluster

from logger import Logger


class CassandraInterface(object):
    def __init__(
        self, ip_address, port=9042, key_space=None, table_name=None, table_schema=None
    ):
        self.ip_address = ip_address if isinstance(ip_address, list) else [ip_address]
        self.port = port
        self.key_space = key_space
        self.key_space_changed = False
        self.table_name = table_name
        self.table_schema = None
        self.session = None

        self.logger = Logger(self.__class__.__name__).get()

    @property
    def key_space(self):
        return self.__key_space

    @key_space.setter
    def key_space(self, key_space):
        self.key_space_changed = True
        self.__key_space = key_space

    @property
    def table_name(self):
        return self.__table_name

    @table_name.setter
    def table_name(self, table_name):
        self.__table_name = table_name

    @property
    def table_schema(self):
        return self.table_schema

    @table_schema.setter
    def table_schema(self, table_schema):
        self.__table_schema = table_schema

    def _connect_to_db(self):
        """
        Achieve a connection to the cassandra db, private method
        :return: A cassandra DB session object
        """
        cluster = Cluster(self.ip_address, self.port)
        session = cluster.connect(self.key_space)
        self.logger.info(
            f"Successfully connected to cluster with {self.key_space if self.key_space else 'no'} keyspace"
        )
        return session

    def connect_to_db(self):
        """
        Public interface (crude singleton) for database connection with Cassandra DB
        :return: A cassandra DB session object
        """
        if not self.session:
            self.session = self._connect_to_db()
        if self.key_space_changed:
            self.session.set_keyspace(self.key_space)
        return self.session

    def retrieve_with_timestamps(
        self, start_timestamp, end_timestamp, remove_tzinfo=True, time_column="key"
    ):
        """
        Make a cql selection query in the cassandra
        remove_tzinfo if set to true will convert the retrieved values to local timezone
        :param start_timestamp: start timestamp as datetime.datetime() object
        :param end_timestamp: end timestamp as datetime.datetime() object
        :param remove_tzinfo: if true then remove the timezone info
        :param time_column: Name of the datetime column
        :return: pd.DataFrame with sorted dates exclusive of both timestamps
        """
        session = self.connect_to_db()
        rows = session.execute(
            f"SELECT * FROM {self.table_name} WHERE {time_column}<'{end_timestamp}' "
            f"and {time_column}>'{start_timestamp}' ALLOW FILTERING;"
        )
        if not rows:
            raise ValueError("No rows were returned from the database")
        df = pd.DataFrame(list(rows))

        if remove_tzinfo:
            df[time_column] = df[time_column].replace(tzinfo=None)
        df.sort_values(by=[time_column], inplace=True)
        self.logger.info(f"Extracted data from {start_timestamp} to {end_timestamp}")
        return df

    def get_initial_data(self, time_column="key"):
        """
        Gets all initial data for training time series models on
        :param time_column: Name of the time column
        :return df: pd.DataFrame() of the Cassandra DB table,
        :return max_time_stamp: Last time stamp of the dataframe
        """
        session = self.connect_to_db()
        rows = session.execute(f"SELECT * FROM {self.table_name}")
        if not rows:
            raise ValueError("No rows were returned from the database")
        df = pd.DataFrame(list(rows))
        max_time_stamp = self.get_last_timestamp(df, time_column=time_column)
        self.logger.info(f"Extracted all data till {max_time_stamp}")
        return df, max_time_stamp

    @staticmethod
    def get_last_timestamp(df, time_column="key"):
        """
        Get the last timestamp existing in the database
        :param df: Dataframe from which last timestamp is to be extracted
        :param time_column: Name of the datetime column
        :return: max(date_index)
        """
        return df[time_column].max().replace(tzinfo=None)

    def write_rows_complete(self):
        pass

    def write_rows_from_timestamp(self, predictions_df):
        """
        Writes rows to Cassandra DB
        :param predictions_df: Predictions with timestamp and identifiers
        :param table_schema: Schema of the Cassandra Table
        :return: None
        """
        session = self.connect_to_db()

        if not self.table_schema:
            self.logger.error("Writing initiated without table schema")
            raise ValueError("Please set the table schema")
        if set(predictions_df.columns) != set(self.table_schema.keys()):
            self.logger.error("Column(s) do not match the give table schema")
            raise ValueError("Column(s) do not match the give table schema")

        col_names = ", ".join(predictions_df.columns)
        query = f"INSERT INTO {self.table_name}({col_names}) VALUES ({', '.join(['?'] * len(predictions_df.columns))});"
        prepared_query = session.prepare(query)
        for row in predictions_df.iterrows():
            row = row[1]
            session.execute(prepared_query, row.values.tolist())

    def _create_key_space(self, new_key_space_name, config_dict=None):
        if not config_dict:
            config_dict = {"class": "SimpleStrategy", "replication_factor": 3}
        query = f"CREATE KEYSPACE IF NOT EXISTS {new_key_space_name} WITH REPLICATION = {str(config_dict)};"
        session = self.connect_to_db()
        try:
            session.execute(query)
        except Exception as e:
            self.logger.error(str(e))
            raise e
        print(
            "Please update the key_space using `object.key_space = new_key_space_name` \n"
            "if you want to start using the created keyspace."
        )
        self.logger.info(f"Successfully created keyspace {new_key_space_name}")

    def _drop_key_space(self, key_space_name):
        if self.key_space == key_space_name:
            raise ValueError(
                f"Keyspace {key_space_name} is in use. Please set it to `None` before trying to drop it."
            )
        query = f"DROP KEYSPACE {key_space_name};"
        self.connect_to_db().execute(query)
        self.logger.info(f"Successfully dropped keyspace {key_space_name}")

    def _create_table(self, new_table_name, schema, primary_key_cols):
        # Create a schema string like 'key timestamp, id text, value double'
        schema_string = ", ".join(
            [
                "".join([column_name, " ", column_type])
                for column_name, column_type in schema.items()
            ]
        )
        # Add which keys are primary keys to the schema string
        add_primary_keys = (
            schema_string + f", PRIMARY KEY ({', '.join(list(primary_key_cols))})"
        )
        query = f"CREATE TABLE IF NOT EXISTS {new_table_name} ({add_primary_keys})"
        self.connect_to_db().execute(query)
        self.logger.info(f"Successfully created table {new_table_name}")
