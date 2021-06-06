import time
from constants import ROOT_DIR

def day_of_week(raw_data, features, auxiliaries, is_inference):
    """
    Extract day of the week (categorical) from tweet_timestamp. UTC time zone

    NOTE: koalas has implemented function to return weekday as number. I guess this suggests wrong ordering,
          being Sunday closer to Monday than Friday. Verify effect on model's predictions
    """
    return raw_data["tweet_timestamp"].apply(lambda x: time.strftime("%A", time.localtime(int(x))))