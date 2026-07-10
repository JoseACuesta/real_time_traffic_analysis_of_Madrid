from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


def initialise_rfr_streaming():
    """
    Initializes a Spark session to simulate streaming data processing.
    This function sets up a local Spark session, defines a schema for the input data,
    reads CSV files in streaming mode from the 'data/RandomForestRegressor/file_sink/' directory, and writes
    the results to the console in continuous update mode.
    :raises pyspark.sql.utils.StreamingQueryException: If an error occurs during streaming execution.
    """

    spark = (
        SparkSession.builder.appName("Simulate Streaming")
        .master("local[8]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("y_test", FloatType(), True),
            StructField("y_pred", FloatType(), True),
        ]
    )

    df = (
        spark.readStream.option("header", True)
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .csv("data/RandomForestRegressor/file_sink/")
    )

    query = df.writeStream.outputMode("update").format("console").start()

    query.awaitTermination()


def initialise_xgboost_streaming():
    """
    Initializes a Spark session to simulate streaming data processing.
    This function sets up a local Spark session, defines a schema for the input data,
    reads CSV files in streaming mode from the 'data/XGBoost/file_sink/' directory, and writes
    the results to the console in continuous update mode.
    :raises pyspark.sql.utils.StreamingQueryException: If an error occurs during streaming execution.
    """

    spark = (
        SparkSession.builder.appName("Simulate Streaming")
        .master("local[8]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("y_test", FloatType(), True),
            StructField("y_pred", FloatType(), True),
        ]
    )

    df = (
        spark.readStream.option("header", True)
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .csv("data/XGBoost/file_sink/")
    )

    query = df.writeStream.outputMode("update").format("console").start()

    query.awaitTermination()
