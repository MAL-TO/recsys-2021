import pandas as pd
import numpy as np

#########################
INPUT_PATH = './sample2m'
OUTPUT_PATH = './sample2m.parquet'
#########################

def load_dataframe(file_name):
  all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                  "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
                "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
                "engaging_user_account_creation", "engagee_follows_engager"]

  all_features_to_idx = dict(zip(all_features, range(len(all_features))))
  labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

  df = pd.read_parquet(
      file_name,
      sep='\x01',
      header=None,
      names=(list(all_features_to_idx.keys()) + list(labels_to_idx.keys())),
      dtype={
        "text_ tokens": np.str,
        "hashtags": np.str,
        "tweet_id": np.str,
        "present_media": np.str,
        "present_links": np.str,
        "present_domains": np.str,
        "tweet_type": "category",
        "language": np.str,
        "tweet_timestamp": np.str,

        "engaged_with_user_id": np.str,
        "engaged_with_user_follower_count": np.int,
        "engaged_with_user_following_count": np.int,
        "engaged_with_user_is_verified": np.bool,
        "engaged_with_user_account_creation": np.str,

        "enaging_user_id": np.str,
        "enaging_user_follower_count": np.int,
        "enaging_user_following_count": np.int,
        "enaging_user_is_verified": np.bool,
        "enaging_user_account_creation": np.str,
        "engagee_follows_engager": np.bool,

        "reply_timestamp": np.str,
        "retweet_timestamp": np.str,
        "retweet_with_comment_timestamp": np.str,
        "like_timestamp": np.str
      },
      parse_dates=False,
  )
  for col in ['text_ tokens', 'hashtags', 'present_media', 'present_links',
              'present_domains']:
    df[col] = df[col].str.split()

  for col in [
    "tweet_timestamp", "reply_timestamp", "retweet_timestamp",
    "retweet_with_comment_timestamp", "like_timestamp",
    "engaged_with_user_account_creation", "engaging_user_account_creation"]:
      df[col] = pd.to_datetime(df[col], unit='s')
  
  return df

if __name__ == '__main__':
  df = load_dataframe(INPUT_PATH)
  df.to_parquet(OUTPUT_PATH)