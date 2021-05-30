import time
import databricks.koalas as ks
import pickle as pkl
import numpy as np
from collections import defaultdict
from functools import reduce
from constants import ROOT_DIR

PATH_AUXILIARIES = os.path.join(ROOT_DIR, "../data/auxiliary")

def user_activity(raw_data, features = None, auxiliary_dict = None):
    """
    Args:
        raw_data (ks.DataFrame): dataset to process for feature extraction
    Returns: 
        new_features (Dict[ks.Series]): Each ks.Series in the dictionary is the counter of user activities
        (appearences as engaging or engagee) inside one of the specified time windows
    """
    def counter_initialization(windows):
        return {k:0 for k in WINDOWS}
    
    def clean_window_counter(window_counter, user):
        # Remove user if its counters are all 0
        if reduce(lambda a, b: a+b, window_counter[user].values()) == 0:
            del window_counter[user]
    
    output_path = os.path.join(self.path_auxiliaries, 'hashtag_window_counter.pkl')
        
    # Time windows in seconds
    WINDOWS = np.array([5, 60, 240, 480, 1440])*60
    j = {k:0 for k in WINDOWS} # Indices to clean up window_counter dictionary when a sample is out of window
    
    if os.exists(output_path):
        with open(OUTPUT_PATH, 'rb') as f:
            initial_dictionary = pkl.load(f)
        window_counter = defaultdict(lambda : counter_initialization(WINDOWS), initial_dictionary)
    
    # Else if training
    else:
        window_counter = defaultdict(lambda : counter_initialization(WINDOWS))
    
    # Sort by timestamp
    raw_data.sort_values(by='tweet_timestamp', inplace = True)
    
    # We convert our to separate numpy array, sicne koalas indexing turns out to be extremely slow
    index_col = raw_data.index.to_numpy()
    engaged_users = raw_data['engaged_with_user_id'].to_numpy()
    timestamps = raw_data['tweet_timestamp'].to_numpy()
    
    new_features = {k:[] for k in WINDOWS}
    for idx, engaged, now in zip(index_col, engaged_users, timestamps):
        tweet_id = idx[0]
        engaging = idx[1]
        
        for time_win in WINDOWS:
            # Remove outdated counts from windows_counter
            while timestamps[j[time_win]] < (now - time_win):
                user_a = index_col[j[time_win]][1]
                user_b = engaged_users[j[time_win]]
                
                if window_counter[user_a][time_win] > 0:
                    window_counter[user_a][time_win] -= 1
                if window_counter[user_b][time_win] > 0:
                    window_counter[user_b][time_win] -= 1
                
                # Remove a user if all of its counter are 0
                clean_window_counter(window_counter, user_a)
                clean_window_counter(window_counter, user_b)
                
                j[time_win] += 1
                
            # Generate new features for current row, and increment window counter by 1
            new_features[time_win].append({
                'tweet_id': tweet_id,
                'engaging_user_id': engaging,
                f'interactions_{time_win}': window_counter[engaging][time_win]
            })
            window_counter[engaging][time_win] += 1
            window_counter[engaged][time_win] += 1
            
    # store current window_counter, since this will be the initial counter at inference time
    with open(OUTPUT_PATH, 'wb') as f:
        pkl.dump(dict(window_counter), f, protocol=pkl.HIGHEST_PROTOCOL)
       
    # Convert each list of dict to a series
    for key in new_features.keys():
        new_features[key] = ks.DataFrame(new_features[key]).set_index(['tweet_id', 'engaging_user_id']).squeeze()

    # store current window_counter, since this will be the initial counter at inference time
    with open(output_path, 'wb') as f:
        pkl.dump(dict(window_counter), f, protocol=pkl.HIGHEST_PROTOCOL)
        
    return new_features