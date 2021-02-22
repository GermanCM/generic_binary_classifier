#%%[markdown]
# Generic imports for logging
# %%
from modelling import binary_classifier as ds_classifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
#%%
from dataset_preprocessing import preprocessing as ds_prep
from tqdm import tqdm
from dataset_EDA import eda as ds_eda
from dataset_elt import dataset_extraction as ds_ext
import logging
import pandas as pd 

logging.basicConfig(filename='classifier.log',
                    level=logging.INFO, 
                    format='%(asctime)s %(message)s')
logger = logging.getLogger("Dataset_eda")

# %%[markdown]
# Load dataset:
dataset_location = r'.\datasets\precipitations_df.csv'

ds_extractor = ds_ext.Dataset_extraction(dataset_location)

columns_names = ['tmp0', 'tmp1', 'hPa', 'hum', 'pp']
limit_rows_number = 5000
# csv mode
ds_extractor.dataset_location = dataset_location
precipitations_df = ds_extractor.load_dataset(csv_mode=True, separator=',')
if precipitations_df is None:
    precipitations_df = pd.read_csv('./datasets/precipitations_df.csv')

#%%
precipitations_df.drop(['Unnamed: 0'], axis=1, inplace=True) 
precipitations_df.head()

# %%[markdown]
# Exploratory Data Analysis (EDA) steps:
"""
    - a first view on the dataframe content: length, some of the first and
      last rows
    - generation of a profile report in HTML format, containing exploratory
      analysis info
      likeattributes correlations, descriptive statistics values, Pearson's
      correlation matrix,
      outlier detections, missing values detections...
    - custom functions to get this info per atribute
"""

eda_obj = ds_eda.Dataset_eda(precipitations_df)
df_length, head_df, tail_df = eda_obj.check_dataframe_content()

# %%[markdown]
'''
eda_obj.profile_dataframe(
    output_file_location_name=".\\dataset_EDA\\eda_reports\\ \
                               precipitations_dataset_eda_report.html")
'''                            
# Attributes discarded due to high correlation over a defined threshold: 0,9
rejected_attrs = eda_obj.get_rejected_attributes(correlation_threshold=0.9)
if rejected_attrs is not None:
  precipitations_df.drop(rejected_attrs, axis=1, inplace=True)
precipitations_df.columns

# %%[markdown]
# Check for any missing values
"""
    If there is a row with more than 50% of the attributes with missing values,
    drop the row.
    Otherwise, impute the missing value in that attribute.
"""
row_indexes_to_delete = eda_obj.check_row_indexes_to_delete(0.5)
row_indexes_to_delete
# %%
"""
  And now, we check for each attribute, any possible missing values
"""
attributes_missing_counts_dict = {}
for attribute in tqdm(precipitations_df.columns):
    attribute_missing_sub_df = eda_obj.check_for_missing_values(attribute)
    if attribute_missing_sub_df is not None:
        attributes_missing_counts_dict[attribute] = len(
            attribute_missing_sub_df)

attributes_missing_counts_dict
# %%[markdown]
'''
  Conversion of the target variable to binary values for the binary
  classification:
  - pp = 0, target value = 0
  - pp > 0, target value = 1

  Moreover, we scale the attributes values so that non-tree based models can
  provide accurate results
'''
# %%
ds_preprocessor = ds_prep.Preprocessing(precipitations_df)
attributes_names = precipitations_df.columns[:-1]
target_name = precipitations_df.columns[-1]
ds_bin_classifier = ds_classifier.Binary_classifier(
    precipitations_df, attributes_names, target_name)
# let's make our target attribute binary:
precipitations_df[target_name] = ds_preprocessor.binarize_target_variable(
    precipitations_df[target_name])
precipitations_df[target_name] = precipitations_df[target_name].apply(
    lambda x: np.int(x))

ds_preprocessor = ds_prep.Preprocessing(precipitations_df)

# %%
X_train, X_validation, y_train, y_validation = ds_bin_classifier.split_into_train_validation_sets(0.3)

# %%
X_train_scaled = ds_preprocessor.standard_scaler_transformer(
    X_train, X_train.columns)
X_train = None

#%%
#X_validation_scaled = ds_preprocessor.standard_scaler_transformer(X_validation, X_validation.columns)
import pickle
X_validation_scaled = pickle.load(open("preprocessor_scaler.pickle", "rb"))
X_validation_scaled = X_validation_scaled.transform(X_validation.values) 
X_validation = None

# %%[markdown]
'''
  Training phase.
  We decide a list of desired models to implement in the training process, all
  of them being part of sklearn:
  - Dummy most-frequent classifier
  - Gaussian Naive Bayes classifier
  - Logistic regression classifier
  - Support Vector classifier
'''
#%%
models_and_params = {'DummyClassifier': {'strategy': ['most_frequent']},
                     'GaussianNB': {'var_smoothing': [1e-09, 1e-08, 1e-10]},
                     'LogisticRegression': {'solver': ['liblinear'],
                                            'penalty': ['l1', 'l2'],
                                            'C': [1, 0.1, 0.01]},
                     'SVC': {'C': [1, 0.1, 0.01], 'gamma': ['scale', 'auto'],
                             'class_weight': ['balanced']}}

Dummy_clf = DummyClassifier()
GaussianNB_clf = GaussianNB()
LogisticRegression_clf = LogisticRegression()
SVC_clf = SVC()
models_list = [Dummy_clf, GaussianNB_clf, LogisticRegression_clf, SVC_clf]

#%%
'''
TEST:
import sklearn
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd 

cv_results_df = pd.DataFrame()
best_estimators_dict = {} 
i = 0
for model_name, hyperparams in tqdm(models_and_params.items()):
    model = sklearn.base.clone(models_list[i])

    clf = GridSearchCV(model, hyperparams, cv=10, scoring=['recall','f1', 'roc_auc'], 
                        refit='roc_auc', return_train_score=True) 
    clf.fit(X_train_scaled, y_train.values)
  
    cv_model_results=pd.DataFrame(pd.DataFrame(clf.cv_results_))
    cv_results_df = cv_results_df.append(cv_model_results) 

    best_estimators_dict[model_name] = clf.best_estimator_
    best_estimator = clf.best_estimator_
    i += 1

cv_results_df
'''
#%%
cv_results_df, best_estimators_dict = \
    ds_bin_classifier.select_model_via_grid_search_cv(models_list,
                                                    models_and_params,
                                                    X_train_scaled,
                                                    y_train.values,
                                                    cv_folds=10,
                                                    scoring_metrics=['recall',
                                                                     'f1',
                                                                     'roc_auc'],
                                                    refit_metric='roc_auc')

cv_results_df

#%%
selected_model = best_estimators_dict['LogisticRegression']
pickle.dump(selected_model, open("selected_model.pickle", "wb"))
selected_model_loaded = pickle.load(open("selected_model.pickle", "rb"))
selected_model_loaded.predict(X_validation_scaled[3].reshape(1, -1))

#%%
'''
best_estimator_df, best_estimator_object = ds_bin_classifier.choose_best_estimator(cv_results_df, 
                                                         'mean_test_roc_auc',
                                                         best_estimators_dict)
'''
max_mean_test_roc_auc = cv_results_df['mean_test_roc_auc'].max()
#print('best model: {}'.format(best_estimator_df[ 'mean_test_recall']))
best_model_info = cv_results_df[cv_results_df['mean_test_roc_auc']==max_mean_test_roc_auc]
best_model_info['param_solver']

#%%[markdown]
## TODO: deshacer lo que sigue y tirar del método 'choose_best_estimator'
max_metric_score = cv_results_df['mean_test_roc_auc'].values.max()
max_metric_score_mask = cv_results_df['mean_test_roc_auc']==max_metric_score
best_estimator_df = cv_results_df[max_metric_score_mask]
best_estimator_name = 'SVC' #best_estimator_df.iloc[0].index
best_estimator_object = best_estimators_dict[best_estimator_name]
#%%
best_estimator_object.predict(X_validation_scaled[6].reshape(1, -1))

#%%[markdown]
## Pruebo a reentrenar el predictor sobre todo el train set:
## Supuestamente, el best_estimator ya es devuelto habiéndose reentrenado en todo el train set (o incluyendo el test set tb?)
best_estimator_object_retrained = best_estimator_object.fit(X_train_scaled, y_train)
#%%
y_preds_best_estim = best_estimator_object.predict(X_validation_scaled)
y_preds_best_estim_retrained = best_estimator_object_retrained.predict(X_validation_scaled)

from sklearn.metrics import roc_auc_score

best_estimator_object_roc_auc = roc_auc_score(y_validation, y_preds_best_estim)
best_estimator_retrained_object_roc_auc = roc_auc_score(y_validation, y_preds_best_estim_retrained)

#%%
print('best_estimator_object_roc_auc: ', best_estimator_object_roc_auc)
print('best_estimator_retrained_object_roc_auc: ', best_estimator_retrained_object_roc_auc)

#%%[markdown]
''' Parece que, según se indica en la docu: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html,
sección 'refit', el best_estimator se reentrena sobre todo el train set, y así vemos el mismo roc_auc que reentrenando a posteriori sobre 
todo el train set: de esta forma, no me haría falta reentrenar tras implementar el GridSearchCV''' 
