from pyspark import SparkConf, SparkContext
import databricks.koalas as ks


def create_spark_context(set_memory_conf=True):
    # https://spark.apache.org/docs/latest/configuration.html
    conf = SparkConf()

    if set_memory_conf:
        conf.set("spark.driver.memory", "4g")
        conf.set("spark.driver.maxResultSize", "4g")

        conf.set("spark.executor.memory", "3g")

    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.execution.arrow.enabled", "true")
    conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")

    conf.set("spark.jars.spark.jars.repositories", "https://repos.spark-packages.org/")
    conf.set("spark.jars.packages", "graphframes:graphframes:0.7.0-spark2.4-s_2.11")

    conf.setMaster("local[*]")
    conf.setAppName("Recsys-2021")

    SparkContext(conf=conf).setLogLevel("WARN")

    ks.set_option("compute.default_index_type", "distributed")
