import time
import databricks.koalas as ks

# Hour of day is not meaningful due to time zones
# What can be useful instead is hour_of_day targed encoded with language

def hour_of_day(raw_data, features = None):
    """
    Extract hour of the day (numerical) from tweet_timestamp. UTC timezone
    """
    result = raw_data["tweet_timestamp"].apply(lambda x: time.strftime("%H", time.localtime(int(x)))).astype('int8')
    print(result.describe())
    return raw_data["tweet_timestamp"].apply(lambda x: time.strftime("%H", time.localtime(int(x)))).astype('int8')