# Twitter RecSys Challenge 2021
https://recsys-twitter.com/

## Run an experiment
### Locally
#### Requirements
- Python 3.7.9
    - You can install this Python version (or any other version) easily with `pyenv`. [Install pyenv](https://github.com/pyenv/pyenv#installation) and run `pyenv install 3.7.9`.
- Java 8 for PySpark
    - If you're running on a Debian-based system, use `adoptopenjdk`. [Link to SO](https://stackoverflow.com/questions/57031649/how-to-install-openjdk-8-jdk-on-debian-10-buster) (that'll save you hours of googling).
- Pipenv
    - `pip install pipenv`

#### How to
1. Pull/clone `recsys-2021` on your system
2. Download one of the [sampled datasets](https://github.com/MAL-TO/recsys-2021/discussions/2) and place it into `data/raw/.`
3. `cd` to the repository root
4. `pipenv install -d` (install both develop and default packages from Pipfile)
5. `pipenv shell` (start the shell for the environment associated to the repository)
6. `cd src`
7. `python run.py ../data/raw/<your_dataset_file> <your_model_of_choice> false`

Check `python run.py --help` for information on additional arguments.

### On the cluster
#### Requirements
- Koalas
    - `pip install koalas`

#### How to
1. Pull/clone `recsys-2021` on your system
2. `cd` to the repository root
3. `cd src`
4. `spark-submit --master yarn --deploy-mode cluster run.py <args>` or (if the former does not work) `spark-submit --master local --deploy-mode client run.py <args>`

## Add a custom feature
To create a new custom feature extractor:

1. Add a file inside `src/preprocessor/<category>/your_feature.py`. If there's no appropriate category directory for your new feature, you can create one.
2. You can use this boilerplate code.
    ```python
    import pandas as pd
    import numpy as np
    import databricks.koalas as ks
    from pyspark.sql import SparkSession

    def your_feature(raw_data, features):
        """Description
        """
        
        # your feature extraction code...
        
        return your_feature_series 
    ```
3. Import your feature in `features_store.py`
4. Use your new feature in your preferred model

Inside this function, you can:

- Access *any* feature available in the raw dataset, i.e., all default features, even ones you don't use directly in your model. Default features are accessible through `raw_data`.
- Access any custom feature extracted *before* your custom feature. The available custom features can be accessed by `features` as a python dict with the structure `{feature_name: ks.Series}`.

A single custom feature extractor can extract more than one feature. If you want to extract more than one feature, you should return a Koalas DataFrame.

**Please make sure that your custom feature extractor returns one row for each input dataset row.**

## Add and use an auxiliary dataset extractor

Auxiliary datasets are DataFrames which may be extracted from raw data (e.g., a tabular representation of an engagement graph). These DataFrames are not used directly as features in the model: instead, they can be used by feature extractors as an auxiliary pre-compiled data source.

Auxiliary DataFrames may be extracted with means of a function (see for example `src/preprocessor/graph/auxiliary_engagement_graph.py`). Once extracted, one or more DataFrames are returned under the form of a Python dict. The return format is the following (even if you want to return only one DataFrame):
``` python
return {
    "auxiliary_df_1": ks.DataFrame,
    "auxiliary_df_2": ks.DataFrame,
    ...
}
```

`auxiliary_df_1`, `auxiliary_df_2`, ..., will be available to *all* feature extractors as keys of the Python dict `auxiliaries`, which is passed as an argument to feature extractors. This means that any feature extractor's function must be defined with an additional argument `auxiliaries`, even if they don't use any auxiliary DataFrame. For instance:
```python
def feature_extractor(raw_data, features, auxiliaries):
    ...
```

The auxiliary dataset extractor function must be defined with two arguments, `raw_data` (which may be either training or test data) and `auxiliary_train`, which defaults to `None`.

This is because, at inference time (when running `inference.py`), the materialized auxiliary datasets, i.e., auxiliary DataFrames pre-computed on training data at training time and saved in `data/auxiliary`, may be integrated with additional data available in the test set, if one implements the additional logic required to do that.

If this is the case, one should check if `auxiliary_train` is not None. If so, we are at *inference time*. In this case, `raw_data` is the test data, and `auxiliary_train` is the pre-compiled auxiliary dataset which must be appended (or integrated in some other way) to the newly extracted auxiliary data coming from `raw_data`, i.e., test data.

Please note that this is not mandatory: you can decide not to run any further integration from test data. If this is the case, put all your train extractor code under an if clause such as `if auxiliary_train is None: ...` and, under `else: ...` (i.e., at inference time), return the keys of `auxiliary_train` corresponding to the auxiliary dataframes of interest.

**Important note**: if you decide to upload a submission with an auxiliary data extractor that you choose to extend at inference time, you *must* also upload the auxiliary dataframe from training time to be extended at test time.

To use an auxiliary dataset extractor in a model, add the auxiliary dataset extractor function to `self.enabled_auxiliaries` in the model constructor.

Check `src/preprocessor/graph/auxiliary_engagement_graph.py` to see a working example.

## Useful Commands

#### Testing the run (training) script without Docker

If you want to clean up all preprocessed columns

```shell
tput reset && \
rm -rf data/preprocessed/* && \
time ARROW_PRE_0_15_IPC_FORMAT=1 PYARROW_IGNORE_TIMEZONE=1 python src/run.py \
    data/raw/sample_200k_rows native_xgboost_baseline false
```

If you want to keep all preprocessed columns

```shell
tput reset && \
time ARROW_PRE_0_15_IPC_FORMAT=1 PYARROW_IGNORE_TIMEZONE=1 python src/run.py \
    data/raw/sample_200k_rows native_xgboost_baseline false
```

#### Testing the inference script without Docker

Remember to run `pipenv shell` first.

This requires that you have `data/raw/test` with some `*part*` files inside.
I suggest testing with ~1-10GB of data to check if memory is wasted somewhere

```shell
# Delete the old results_folder, a to_csv artifact
rm -rf results_folder && \
# Clear the terminal (optional)
tput reset && \
# Launch inference.py with the environment variables needed by Apache Arrow
ARROW_PRE_0_15_IPC_FORMAT=1 PYARROW_IGNORE_TIMEZONE=1 python src/inference.py
```
