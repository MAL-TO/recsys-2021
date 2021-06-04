def media_count(raw_data, features = None, auxiliary_dict = None):
    """
    Count number of words in "text_tokens" field of raw data
    """
    return raw_data["present_media"].apply(lambda x: len(x.split("\t")))