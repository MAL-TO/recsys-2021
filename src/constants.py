import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # local
ROOT_DIR_CLUSTER = "file:///home/bigdata-01QYD/sXXXXXX/recsys-2021"

PATH_PREPROCESSED = os.path.join(ROOT_DIR, "../data/preprocessed")
PATH_PREPROCESSED_CLUSTER = os.path.join(ROOT_DIR_CLUSTER, "data/preprocessed")

PATH_AUXILIARIES = os.path.join(ROOT_DIR, "../data/auxiliary")
PATH_AUXILIARIES_CLUSTER = os.path.join(ROOT_DIR_CLUSTER, "data/auxiliary")

MODEL_SEED = None

# List of Python dicts, each dict represents a fold with train and test set
# filenames inside `PATH_DATA`
PATH_DATA = "../data/raw"
FILENAMES_DATA = [
    {"train": "", "test": ""},
]
