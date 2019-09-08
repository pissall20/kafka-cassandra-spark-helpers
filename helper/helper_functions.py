import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import TimestampType


def create_spark_session(master_url, packages=None):
    """
    Creates a local spark session
    :param master_url: IP address of the cluster you want to submit the job to or local with all cores
    :param packages: Any external packages if needed, only when called. This variable could be a string of the package
        specification or a list of package specifications.
    :return: spark session object
    """
    if packages:
        packages = ",".join(packages) if isinstance(packages, list) else packages
        spark = (
            SparkSession.builder.master(master_url)
            .config("spark.io.compression.codec", "snappy")
            .config("spark.ui.enabled", "false")
            .config("spark.jars.packages", packages)
            .getOrCreate()
        )
    else:
        spark = (
            SparkSession.builder.master(master_url)
            .config("spark.io.compression.codec", "snappy")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    return spark


def connect_to_sql(spark_master_url, jdbc_hostname, jdbc_port, database, data_table, username, password):
    spark = create_spark_session(spark_master_url)
    jdbc_url = "jdbc:mysql://{0}:{1}/{2}".format(jdbc_hostname, jdbc_port, database)

    connection_details = {
        "user": username,
        "password": password,
        "driver": "com.mysql.cj.jdbc.Driver",
    }

    df = spark.read.jdbc(url=jdbc_url, table=data_table, properties=connection_details)
    return df


def cassandra_connector(spark_master_url, connection_host, table_name, key_space, cassandra_package=None):
    # A cassandra connector is needed
    if cassandra_package:
        needed_spark_package = cassandra_package
    else:
        needed_spark_package = "com.datastax.spark:spark-cassandra-connector_2.11:2.4.0"

    spark = create_spark_session(
        master_url=spark_master_url, packages=needed_spark_package
    )

    spark.conf.set("spark.cassandra.connection.host", str(connection_host))
    data = spark.read.format("org.apache.spark.sql.cassandra").load(keyspace=key_space, table=table_name)
    return data


def spark_date_parsing(df, date_column, date_format):
    """
    Parses the date column given the date format in a spark dataframe
    NOTE: This is a Pyspark implementation

    Parameters
    ----------
    :param df: Spark dataframe having a date column
    :param date_column: Name of the date column
    :param date_format: Simple Date Format (Java-style) of the dates in the date column

    Returns
    -------
    :return: A spark dataframe with a parsed date column
    """
    # Check if date has already been parsed, if it has been parsed we don't need to do it again
    date_conv_check = [
        x[0] for x in df.dtypes if x[0] == date_column and x[1] in ["timestamp", "date"]
    ]
    """
    # mm represents minutes, MM represents months. To avoid a parsing mistake, a simple check has been added

    The logic here is, if the timestamp contains minutes, the date_format string will be greater than 10 letters. 
    Minimum being yyyy-MM-dd which is 10 characters. In this case, if mm is entered to represent months, 
    the timestamp gets parsed for minutes, and the month falls back to January
    """
    if len(date_format) < 12:
        date_format = date_format.replace("mm", "MM")

    if not date_conv_check:
        df = df.withColumn(date_column, F.to_timestamp(F.col(date_column), date_format))
    # Spark returns 'null' if the parsing fails, so first check the count of null values
    # If parse_fail_count = 0, return parsed column else raise error
    parse_fail_count = df.select(
        ([F.count(F.when(F.col(date_column).isNull(), date_column))])
    ).collect()[0][0]
    if parse_fail_count == 0:
        return df
    else:
        raise ValueError(
            f"Incorrect date format '{date_format}' for date column '{date_column}'"
        )


def find_date_range(df, date_column):
    """
    Finds the minimum and maximum date of a date column in a DF
    :param df: A spark dataframe
    :param date_column: A parsed date column name
    :return: Min and Max of the date column (as a tuple of datetime.datetime() objects)
    """
    # Find the date-range
    dates = df.select([F.min(date_column), F.max(date_column)]).collect()[0]
    min_date, max_date = (
        dates["min(" + date_column + ")"],
        dates["max(" + date_column + ")"],
    )
    return min_date, max_date
