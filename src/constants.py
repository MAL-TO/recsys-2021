import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # local
ROOT_DIR_CLUSTER = "file:///home/bigdata-01QYD/s277596/recsys-2021/src"

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")
PATH_PREPROCESSED_CLUSTER = os.path.join(ROOT_DIR_CLUSTER, "../data/preprocessed")

PATH_AUXILIARIES = os.path.join(ROOT_DIR, "../data/auxiliary")
PATH_AUXILIARIES_CLUSTER = os.path.join(ROOT_DIR_CLUSTER, "../data/auxiliary")

MODEL_SEED = None

# List of Python dicts, each dict represents a fold with train and test set
# filenames inside `PATH_DATA`
PATH_DATA = "../data/raw"
PATH_DATA_CLUSTER = os.path.join(ROOT_DIR_CLUSTER, PATH_DATA)
FILENAMES_DATA = [
    {"train": "sample1", "test": "sample1_test"},
    {"train": "sample2", "test": "sample2_test"},
    {"train": "sample3", "test": "sample3_test"},
    {"train": "sample4", "test": "sample4_test"},
    {"train": "sample5", "test": "sample5_test"},
]
