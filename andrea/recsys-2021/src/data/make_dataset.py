# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    def load_dataframe(file_name):
        all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                    "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
                    "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                    "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
                    "engaging_user_account_creation", "engagee_follows_engager"
                       ]

        all_features_to_idx = dict(zip(all_features, range(len(all_features))))
        labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

        df = pd.read_csv(
            file_name,
            sep='\x01',
            header=None,
            names=(list(all_features_to_idx.keys()) + list(labels_to_idx.keys())),
            dtype={
            "text_ tokens": str,
            "hashtags": str,
            "tweet_id": str,
            "present_media": str,
            "present_links": str,
            "present_domains": str,
            "tweet_type": "category",
            "language": str,
            "tweet_timestamp": str,

            "engaged_with_user_id": str,
            "engaged_with_user_follower_count": int,
            "engaged_with_user_following_count": int,
            "engaged_with_user_is_verified": bool,
            "engaged_with_user_account_creation": str,

            "enaging_user_id": str,
            "enaging_user_follower_count": int,
            "enaging_user_following_count": int,
            "enaging_user_is_verified": bool,
            "enaging_user_account_creation": str,
            "engagee_follows_engager": bool,

            "reply_timestamp": str,
            "retweet_timestamp": str,
            "retweet_with_comment_timestamp": str,
            "like_timestamp": str
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

    df = load_dataframe(input_filepath)
    df.to_parquet(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()