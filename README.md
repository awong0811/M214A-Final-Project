# M214A Final Project
Anthony Wong, Mohammad Alkharafi, Borys Jastrzebski

## Installation
If running locally, you can install the environment using conda:
```bash
$ conda env create -f environment.yml
```
This creates an environment called 'xgb'.
Otherwise, create a python virtual environment and install the packages in the environment.yml file. The specific versions of the packages are listed in the requirements.txt file (also conda-only for direct installation).

## Saved Features
Here's the [link](https://drive.google.com/file/d/1A2Ve2IBPSgJmcVmx1NVWEIX4eGza2Xm4/view?usp=sharing) to a zip folder of saved features that can be used to run the entire notebook without creating new features. Simply load in desired features as shown below. <name> only includes the feature name; does not include '_train', '_test...', etc. The load_features() function returns the training, testing (clean), and testing (noisy) dataframes in that order.
```python
from preprocess import *
feat_dataframes = load_features(<name>)
```

## Running the Notebook/Data Pipeline
The utility functions are in the separate train.py, preprocess.py, and explainability files to improve readability.
The train.py and explainability.py files contain the given code for fitting the model and generating the SHAP charts/confusion matrices respectively.
\
\
Running this should import everything you need to run the data pipeline.
```python
from preprocess import *
from train import *
from explainability import * 
```
\
Running this block of code makes a new set of features based on which feature extraction function you pick and saves the features. If a ./saved_features/ directory does not already exist, it is created and the features are saved in there. A new XGBoost model is fitted and the shap chart/confusion matrices are displayed as well.
```python
name = '<desired_feature_name>'
args_dict = {'<arg>': <arg_val>}
data = train_test_preprocess(<func>, name, args_dict)
model = xgboost.XGBClassifier(tree_method="hist", device="cuda")
train(data, model)
save_features(data, name)
shap_explain(data[0],model)
confusion_matrix(data, model)
```
\
Running this block of code concatenates preexisting sets of saved features instead of creating a new set of features (their .csv files must already exist in some saved_features/ folder). The rest is the same as the above block of code.
```python
name = '<desired_feature_name>'
data = concatenate_features(names=['<feature_1_name>', '<feature_2_name>',])
model = xgboost.XGBClassifier(tree_method="hist", device="cuda")
train(data, model)
save_features(data, name)
shap_explain(data[0],model)
confusion_matrix(data, model)
```
\
Below is a description of the train_test_preprocess() function in preprocess.py. The rest of the functions should be self-explanatory. Refer to the docstrings in the functions for further clarification if needed.

| Arguments | Type | Description |
|-----------|------|-------------|
| func | Python function | Select one of the feature extraction functions in the preprocess.py file from the Feature Extraction section. |
| feat_name | str | Desired name for the features in the dataframes. |
| args | dict | A dictionary with keys and values for the corresponding arguments in the feature extraction function (func). This dictionary will be unpacked and passed into func. See code in notebook for examples. |

## Contact
Email awong0811@ucla.edu if you have any questions.