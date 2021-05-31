import time
import os
import databricks.koalas as ks
import pickle as pkl
from collections import defaultdict
from constants import ROOT_DIR

PATH_AUXILIARIES = os.path.join(ROOT_DIR, "../data/auxiliary")

def hashtag_popularity(raw_data, features = None, auxiliary_dict = None):
    """
    Args:
        raw_data (ks.DataFrame): dataset to process for feature extraction
    Returns: 
        new_feature (ks.DataFrame): DataFrame where for each input sample, we have the corresponding hashtag counter.
    """
    
    WINDOW_SIZE = 7200 # 2 hours time window
    output_path = os.path.join(PATH_AUXILIARIES, 'hashtag_window_counter.pkl')
    
    # Use loc to pass a view instead of a copy
    raw_data.sort_values(by='tweet_timestamp', inplace = True)

    # TODO: verify that order is preserved!
    # Compare 10^6 samples required time, with 10k samples, which is about 6 seconds
    # Conversion to numpy is done since dataframe indexing is complicated and unfeasibly slow (many hours vs few seconds...)
    indices = raw_data.index.to_numpy()
    hashtags = raw_data['hashtags'].to_numpy()
    timestamps = raw_data['tweet_timestamp'].to_numpy()

    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamps[0])))
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamps[-1])))

    
    # Initialize with existing dict at inference time (same for training without first chunck of data)
    if os.exists(output_path):
        with open(OUTPUT_PATH, 'rb') as f:
            initial_dictionary = pkl.load(f)
        window_counter = defaultdict(lambda : 0, initial_dictionary)
    
    # Else if training
    else:
        window_counter = defaultdict(lambda : 0)
    
    # pointer to reduce counter when timestamp < now - WINDOW_SIZE
    j = 0
    
    new_col = []
    for idx, hash, timestamp in zip(indices, hashtags, timestamps): # RUNTIME: O(N)
        hashtags = hash.split('\t') # list of hashtags whose counter must be incremented
        now = timestamp # last tweet timestamp
        
        # Remove hashtags out of the 2 hours time window from now
        while timestamps[j] < (now - WINDOW_SIZE):
            hashtags_to_decrement = hashtags[j]
            for h in hashtags_to_decrement:
                window_counter[h] -= 1
                if window_counter[h] <= 0:
                    del window_counter[h]
                
            j += 1
        
        
        # I have more than one hashtag for each record, but i need only one counter
        # Need to make a summary of the hashtags ==> max of window_counter?           
        if hashtags[0]:
            most_popular = 0
            for h in hashtags:
                window_counter[h] += 1
                if window_counter[h] > most_popular:
                    most_popular = window_counter[h]
                
                # Increment corresponding hashtag counter
                # window_counter[h] += 1

            # Index is tuple for multiIndex. Ulima cosa da fare questo!
            new_col.append({
                'tweet_id': idx[0],
                'engaging_user_id': idx[1],
                'counter': most_popular
            })
        
        else:
            # How to represent missing hashtag?
            new_col.append({
                'tweet_id': idx[0],
                'engaging_user_id': idx[1],
                'counter': -1
            })
    
    new_feature = ks.DataFrame(new_col).set_index(['tweet_id', 'engaging_user_id']).squeeze()
    
    # store current window_counter, since this will be the initial counter at inference time
    with open(output_path, 'wb') as f:
        pkl.dump(dict(window_counter), f, protocol=pkl.HIGHEST_PROTOCOL)

    return new_feature