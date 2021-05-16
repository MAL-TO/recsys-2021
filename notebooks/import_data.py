import pandas as pd


column_dtypes = {
    'text_tokens': str,
    'hashtags': str,
    'tweet_id': str,
    'present_media': str,
    'present_links': str,
    'present_domains': str,
    'tweet_type': str,
    'language': str,
    'tweet_timestamp': int,

    'engaged_with_user_id': str,
    'engaged_with_user_follower_count': int,
    'engaged_with_user_following_count': int,
    'engaged_with_user_is_verified': bool,
    'engaged_with_user_account_creation': int,

    'engaging_user_id': str,
    'engaging_user_follower_count': int,
    'engaging_user_following_count': int,
    'engaging_user_is_verified': bool,
    'engaging_user_account_creation': int,

    'engagee_follows_engager': bool,

    'reply_timestamp': 'Int64',
    'retweet_timestamp': 'Int64',
    'retweet_with_comment_timestamp': 'Int64',
    'like_timestamp': 'Int64'
}
categorical_features = [
    'language',
    'tweet_type',
    'present_media',
]

timestamp_features = ['tweet_timestamp', 'engaged_with_user_account_creation', 'engaging_user_account_creation']

targets = ["reply", "retweet", "retweet_with_comment", "like"]

def import_data(path):
    df = pd.read_csv(
        path,
        names=list(column_dtypes.keys()),
        sep='\x01',
        dtype=column_dtypes)
    df = df.assign(**{
        "reply": df["reply_timestamp"].notna(),
        "retweet": df["retweet_timestamp"].notna(),
        "retweet_with_comment": df["retweet_with_comment_timestamp"].notna(),
        "like": df["like_timestamp"].notna()
    }).drop([
        'text_tokens',
        'hashtags',

        'present_links',
        'present_domains',
        
        'engaged_with_user_id',
        'engaging_user_id',

        'reply_timestamp',
        'retweet_timestamp',
        'retweet_with_comment_timestamp',
        'like_timestamp'
    ], axis=1)
    
    df['present_media'] = df['present_media'].fillna(value="None")
    
    for f in timestamp_features:
        df[f] = pd.to_datetime(df[f], origin='unix', unit='s')
    
    for f in categorical_features:
        df[f] = df[f].astype('category')
    
    df = df.set_index('tweet_timestamp', drop=False).sort_index()

    assert df.notna().all().all(), "the data must not contain NAs"
    
    return df