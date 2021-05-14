import time

def day_of_week(raw_data, features = None):
    """
    Extract hour of the day (numerical) from tweet_timestamp. UTC timezone
    """
    return ks["tweet_timestamp"].apply(lambda x: time.strftime("%H", x))