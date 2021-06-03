def binarize_timestamps(raw_data, features, auxiliaries):
    return {
        "reply": raw_data["reply_timestamp"].notna(),
        "retweet": raw_data["retweet_timestamp"].notna(),
        "retweet_with_comment": raw_data["retweet_with_comment_timestamp"].notna(),
        "like": raw_data["like_timestamp"].notna(),
    }