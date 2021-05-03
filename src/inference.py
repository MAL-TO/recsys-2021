import os

from constants import ROOT_DIR
from data.importer import import_data
from preprocessor.features_store import FeatureStore
from model.h2o_xgboost_baseline import Model

PATH_PREPROCESSED = os.path.join(ROOT_DIR, '../data/preprocessed')

def main():
    print("Initializing model...", end = " ")
    model = Model(include_targets=False)
    model.load_pretrained()
    print("Done")

    print("Importing data...", end=" ")
    raw_data = import_data(os.path.join(ROOT_DIR, '../test'), include_targets=False)
    print("Done")

    print("Assembling dataset...", end=" ")
    store = FeatureStore(PATH_PREPROCESSED, model.enabled_features, raw_data, is_cluster=False)
    features_union_df = store.get_dataset()
    print("Dataset ready")

    print('Computing predictions...', end=" ")
    predictions_df = model.predict(features_union_df)
    print("Done")

    assert len(features_union_df) == len(predictions_df), \
        "features and predictions must have the same length"

    predictions_df['tweet_id'] = raw_data['tweet_id'].to_numpy()
    predictions_df['user_id'] = raw_data['engaging_user_id'].to_numpy()

    predictions_df[
        ['tweet_id', 'user_id', 'reply', 'retweet', 'retweet_with_comment', 'like']
    ].to_csv('results.csv', mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
