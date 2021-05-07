# recsys-2021
https://recsys-twitter.com/

## Run an experiment
### Locally
#### Requirements
- Python 3.7.9
    - You can install this Python version (or any other version) easily with `pyenv`. [Install pyenv](https://github.com/pyenv/pyenv#installation) and run `pyenv install 3.7.9`.
- Java 8 for PySpark
    - If you're running on a Debian-based system, use `adoptopenjdk`. [Link to SO](https://stackoverflow.com/questions/57031649/how-to-install-openjdk-8-jdk-on-debian-10-buster). (that'll save you hours of googling).
- Pipenv
    - `pip install pipenv`

#### How to
1. Pull/clone `recsys-2021` on your system
2. `cd` to the repository
3. `pipenv install -d` (install both develop and default packages from Pipfile)
4. `pipenv shell` (start the shell for the environment associated to the repository)
5. `cd src`
6. Make sure that `dataset_name` is set to the corresponding path in `RAW_DATA_INPUT_PATH`, and that the corresponding path is correct.
7. `python run.py`

### On the cluster
#### Requirements
- Koalas
    - `pip install koalas`

#### How to
1. Pull/clone `recsys-2021` on your system
2. `cd` to the repository
3. `cd src`
4. Make sure that `dataset_name` is set to the corresponding path in `RAW_DATA_INPUT_PATH`, and that the corresponding path is correct.
5. `spark-submit --master yarn --deploy-mode cluster run.py` or (if the former does not work) `spark-submit --master local --deploy-mode client run.py`

## Add a custom feature
To create a new custom feature extractor:

1. Add your new feature `your_feature` under the `"custom"` object inside `config.json`. 
2. Add a file inside `src/preprocessor/<category>/your_feature.py` (if there's no appropriate category directory for your new feature, you can create one). Note that the python file **must** have the same name as the feature added inside `config.json`.
3. Copy-paste this boilerplate code:
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

4. Change `your_feature` to match **exactly** the filename (except the .py extension).

Inside this function, you can:

- Access *any* feature available in the raw dataset, i.e., all default features, even the ones disabled in `config.json`. Default features are accessible through `raw_data`.
- Access any custom feature extracted *before* your custom feature, i.e., any feature that comes before it inside `config.json`. Custom features previously extracted are available through `features` as a python dict with the structure `{feature_name: ks.Series}`. Place your new feature inside `config.json` according to your needs.

A single custom feature extractor can extract more than one feature. If you want to extract more than one feature, you should return a python dictionary with the following structure:

```sql
{
	"my_feature": ks.Series,
	"another_feature", ks.Series,
	...
	"last_entry": ks.Series
}
```

**Please make sure that your custom feature extractor returns one row for each input dataset row.**

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
