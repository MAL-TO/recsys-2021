from transformers import BertTokenizer

def bert_test(raw_data, features = None):
    """
    Test function to detect @ characters in a tweet, form bert tokens ID.
    The goal is to get a function able to understand if what are the most frequent person tagged by a user.

    Once we have this information, we check if engaging is among this people ==> basically
    the goal is to get a binary feature saying: is engaging frequently tagged by engagee?

    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_tokens = raw_data['text_tokens']

    # TODO(Francesco): If you want to retain only the most appearing, you need
    # 1. to remove stopwords
    # 2. to decode only the most common, since it is faster
    tweets_strings =  text_tokens.apply(lambda tokens_list: tokenizer.decode(map(lambda t: int(t), tokens_list.split("\t"))))

    for _, tweet in tweets_strings.iteritems():
        tokens = tweet.split(" ")
        for t in tokens:
            if '@' in t:
                print(t)

    return None