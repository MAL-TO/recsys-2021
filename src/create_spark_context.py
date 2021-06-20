from pyspark import SparkConf, SparkContext
import databricks.koalas as ks


def create_spark_context(set_memory_conf=True, h2o=True):
    # https://spark.apache.org/docs/latest/configuration.html
    conf = SparkConf()

    if set_memory_conf:
        conf.set("spark.driver.memory", "100g")
        conf.set("spark.driver.maxResultSize", "100g")

        conf.set("spark.executor.memory", "100g")

    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.execution.arrow.enabled", "true")
    conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")

    conf.set("spark.jars.spark.jars.repositories", "https://repos.spark-packages.org/")
    conf.set("spark.jars.packages", "graphframes:graphframes:0.7.0-spark2.4-s_2.11")

    if h2o:
        # http://docs.h2o.ai/sparkling-water/2.4/latest-stable/doc/configuration/configuration_properties.html
        conf.set("spark.dynamicAllocation.enabled", "false")
        conf.set("spark.ext.h2o.log.level", "WARN")

    conf.setMaster("local[*]")
    conf.setAppName("Recsys-2021")

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    ks.set_option("compute.default_index_type", "distributed")

    return sc
