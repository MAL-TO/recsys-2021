from pyspark import SparkConf, SparkContext
import databricks.koalas as ks


def create_spark_context():
    # https://spark.apache.org/docs/latest/configuration.html
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.driver.maxResultSize", "4g")

    conf.set("spark.executor.memory", "3g")

    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.execution.arrow.enabled", "true")
    conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")

    conf.setMaster("local[*]")
    conf.setAppName("Recsys-2021")

    SparkContext(conf=conf).setLogLevel("WARN")

    ks.set_option("compute.default_index_type", "distributed")
