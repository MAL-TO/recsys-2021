import koalas as ks
import pickle as pkl
from collections import defaultdict

def hashtag_popularity(raw_data, features = None):
    """
    Args:
        raw_data (ks.DataFrame): dataset to process for feature extraction
    Returns: 
        new_feature (ks.DataFrame): DataFrame where for each input sample, we have the corresponding hashtag counter.
    """
    
    WINDOW_SIZE = 7200 # counter refers to last 7200 seconds = 2 hours
    OUTPUT_PATH = 'hashtag_window_counter.pkl' # path to store window counter. Will be used for inference
    
    # Use loc to pass a view instead of a copy
    hash_time_df = raw_data.loc[:, ('hashtags', 'tweet_timestamp')]
    hash_time_df.sort_values(by='tweet_timestamp', inplace = True)
    
    # Add initialization for existing dictionary at inference time
    # Same for training if we drop first chunk of data
    window_counter = defaultdict(lambda : 0) # initialize to 0 if key is not present
    
    # number of active hashtags, needed for normalization
    # do we want to normalize? I would not, in order to actually detect hot hashtag,
    # and this is contained in absolute count, not relative
    active_counter = 0 
    
    # pointer to reduce counter when timestamp < now - WINDOW_SIZE
    j = 0
    
    # new column container
    new_col = []
    for index, row in hash_time_df.iterrows():
        hashtags = row['hashtags'].split('\t') # list of hashtags whose counter must be incremented
        now = int(row['tweet_timestamp']) # last tweet timestamp
        
        # Remove hashtags out of the 2 hours time window form now
        while int(hash_time_df.iloc[j, -1]) < (now - WINDOW_SIZE):
            row_to_delete = hash_time_df.iloc[j]
            hashtags_to_decrement = row_to_delete['hashtags']
            for h in hashtags_to_decrement:
                window_counter[h] -= 1
                if window_counter[h] <= 0:
                    del window_counter[h]
                active_counter -= 1
                
            j += 1
        
        # I have more than one hashtag for each record, but i need only one counter
        # Need to make a summary of the hashtags ==> max of window_counter?                
        most_popular = 0
        for h in hashtags:
            if h:  # h not empty string ''
                if window_counter[h] > most_popular:
                    most_popular = window_counter[h]
                
                # Increment corresponding hashtag counter
                window_counter[h] += 1
                active_counter += 1

        # Index is tuple for multiIndex. Ulima cosa da fare questo!
        new_col.append({
            'tweet_id': index[0],
            'engaging_user_id': index[1],
            'counter': most_popular
        })
            
    new_feature = ks.DataFrame(new_col).set_index(['tweet_id', 'engaging_user_id'])
    
    # store current window_counter, since this will be the initial counter at inference time
    with open(OUTPUT_PATH, 'wb') as f:
        pkl.dump(dict(window_counter), f, protocol=pkl.HIGHEST_PROTOCOL)

    return new_feature