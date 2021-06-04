import time

# NVIDIA
# TODO(Francesco): test
def word_count(raw_data, features = None, auxiliary_dict = None):
    """
    Count number of words in "text_tokens" field of raw data
    """
    # TODO(test) try stopwords removal (maybe expensive and ineffective)
    return raw_data["text_tokens"].apply(lambda x: len(x.split("\t")))