# Databricks notebook source
import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import mlflow

class Preprocessing():
    """
        This class is ready to receive an existing 'dataframe'
        to implement preprocessing actions
    """
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def read_csv(self, dataset_location, cols_to_use, sep_val=','):
        try:
            self.dataframe = pd.read_csv(dataset_location, usecols=cols_to_use, sep=sep_val)
            return self.dataframe
          
        except Exception as exc:
            return exc

    def impute_attribute(self, attribute, desired_strategy):
        try:
            from sklearn.preprocessing import Imputer
            import numpy as np

            attribute_values = self.dataframe[attribute].values.reshape(-1, 1)
            imp = Imputer(missing_values=np.nan, strategy=desired_strategy,
                          axis=0)
            imp.fit(attribute_values)
            transformed_values = imp.transform(attribute_values)
            self.dataframe.loc[:, attribute] = transformed_values
            return self.dataframe

        except Exception as exc:
            return exc

    def binarize_target_variable(self, target_series):
        try:
            binary_target_series_mask = target_series > 0
            target_series[binary_target_series_mask] = 1
            return target_series

        except Exception as exc:
            return exc

    def standard_scaler_transformer(self, attributes_set, columns_to_scale, preprocessor_saving_path="/preprocessor/preprocessor_scaler.pickle"):
        try:
            from sklearn.preprocessing import StandardScaler
            import pandas as pd
            from sklearn.pipeline import Pipeline
            import pickle 

            scaler = StandardScaler()
            scaler_pipeline = Pipeline([('scaler', scaler)])

            #scaled_df_values = scaler.fit_transform(attributes_set.values)
            fitted_scaler = scaler_pipeline.fit(attributes_set.values)
            pickle.dump(fitted_scaler, open(preprocessor_saving_path, "wb"))

            scaled_df_values = fitted_scaler.transform(attributes_set.values)
            scaled_df = pd.DataFrame(columns=columns_to_scale,
                                     data=scaled_df_values)
            
            return scaled_df

        except Exception as exc:
            print(exc)
            return exc

    def get_rejected_attributes(self, correlation_threshold=0.9):
        try:
            import pandas_profiling

            array_df_profile = pandas_profiling.ProfileReport(self.dataframe)
            array_df_rejected = array_df_profile.get_rejected_variables(correlation_threshold)
            
            return array_df_rejected

        except Exception as exc:
            return exc
          
    def check_values_range(self, attribute, expected_range_min, expected_range_max):
        try:
            below_range_mask = self.dataframe[attribute].values < expected_range_min
            above_range_mask = self.dataframe[attribute].values > expected_range_max
            below_range_samples = self.dataframe[below_range_mask]
            above_range_samples = self.dataframe[above_range_mask]

            return below_range_samples, above_range_samples
        except Exception as exc:
            return exc

    def get_outliers_tukey_test(self, attribute):
        """[summary]
        
        Arguments:
            attribute {string} -- dataframe column on which we can checks possible outliers
        
        Returns:
            list, dataframe -- outlier indices list, subdataframe containing the samples with found outlier
        """
        try:
            import numpy as np 
            mask_not_nan_values = np.isnan(self.dataframe[attribute].values)==False
            not_nan_values = self.dataframe[attribute][mask_not_nan_values]

            q1 = np.percentile(not_nan_values, 25)
            q3 = np.percentile(not_nan_values, 75)
            iqr = q3 - q1 
            
            floor = q1 - 1.5*iqr
            ceiling = q3 + 1.5*iqr
            
            outlier_indices = list(self.dataframe[attribute].index[(self.dataframe[attribute].values < floor)|(self.dataframe[attribute].values > ceiling)])  
            samples_with_outlier_values = self.dataframe.iloc[outlier_indices]

            return outlier_indices, samples_with_outlier_values
        except Exception as exc:
            return exc
          
    def check_for_missing_values(self, attribute):
        """Computes the sub dataframe containing missing values on the defined attribute

            Parameters
            ----------
            attribute : column name which to check for missing values on

            Returns
            -------
            sub dataframe containing the attribute missing values
        """
        try:
            import numpy as np
            nan_values_mask = np.isnan(self.dataframe[attribute])==True
        
            return self.dataframe[nan_values_mask] 

        except Exception as exc:
            return exc

    def check_row_indexes_to_delete(self, missing_threshold=0.5):
        try:
            not_missing_counts = self.dataframe.count(axis='columns')
            no_missing_rates = not_missing_counts/len(self.dataframe.columns)
            missing_rate_over_threshold_mask = no_missing_rates < missing_threshold 

            row_indexes_to_delete = no_missing_rates[missing_rate_over_threshold_mask].index
            return row_indexes_to_delete

        except Exception as exc:
            return exc
            

class SBS():
    """
       Sequential Backward Selection: selects the best 'k_features' attributes,
       based on a defined metric and using a defined estimator.
       Credits: Sebastian Raschka
       More info:
       http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    """
    from sklearn.base import clone
    from itertools import combinations
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test,
                                 self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                best = np.argmax(scores)
                self.indices_ = subsets[best]
                self.subsets_.append(self.indices_)
                dim -= 1
                self.scores_.append(scores[best])
                self.k_score_ = self.scores_[-1]
                return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

class Binary_classifier():
    """
        - receives a dataframe, together with the attributes and target names, validation fraction and optional random state
        - with this info, implements a model and hyperparameters selection via grid search cross-validation among a given list 
          of desired algorithms candidates   
        - all the final model candidates must improve a dummy baseline model's performance
    """    
    def __init__(self, dataframe, attributes, target, validation_fraction=0.3, random_state=42):
        self.dataframe = dataframe
        self.attributes = attributes
        self.target = target
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.best_estimator = None

    def split_into_train_validation_sets(self, test_fraction):
        """Splits the dataset into a single train-test split

            Parameters
            ----------
            test_fraction : test fraction

            Returns
            -------
            X_train(train attributes values), X_validation(validation attributes values), 
            y_train(train target values), y_validation(validation target values)
        """
        try:
            from sklearn.model_selection import train_test_split

            X_train, X_validation, y_train, y_validation = train_test_split(self.dataframe[self.attributes], self.dataframe[self.target], \
                        test_size=test_fraction, stratify=self.dataframe[self.target].values, random_state=self.random_state)

            return X_train, X_validation, y_train, y_validation
        
        except Exception as exc:
            return exc

    def select_model_via_grid_search_cv(self, models_list, models_params_dict, X_train, y_train, cv_folds=5, scoring_metrics=['recall', 'f1', 'roc_auc'], refit_metric='roc_auc'):
        try:
            import sklearn
            from sklearn.model_selection import GridSearchCV
            from tqdm import tqdm
            import pandas as pd 

            cv_results_df = pd.DataFrame()
            best_estimators_dict = {} 
            i = 0
            for model_name, hyperparams in tqdm(models_params_dict.items()):
                with mlflow.start_run(run_name='rain_forecaster_{}'.format(model_name), nested=True):
                    mlflow.set_tag('use_case_name', 'rain_forecaster')
                    mlflow.set_tag('model_type', model_name)
                    
                    model = sklearn.base.clone(models_list[i])

                    clf = GridSearchCV(model, hyperparams, cv=cv_folds, scoring=scoring_metrics, 
                                       refit=refit_metric, return_train_score=True) 
                    clf.fit(X_train, y_train)

                    cv_model_results=pd.DataFrame(pd.DataFrame(clf.cv_results_))
                    cv_results_df = cv_results_df.append(cv_model_results) 

                    best_estimators_dict[model_name] = clf.best_estimator_
                    self.best_estimator = clf.best_estimator_

                    mlflow.sklearn.log_model(clf.best_estimator_, "rain_forecaster")
                    i += 1

            return cv_results_df, best_estimators_dict
        
        except Exception as exc:
            return exc

    def choose_best_estimator(self, validation_results_dataframe, metric, 
                              best_estimators_dictionary):
        try:
            max_metric_score = validation_results_dataframe[metric].values.max()
            max_metric_score_mask = validation_results_dataframe[metric]==max_metric_score
            best_estimator_df = validation_results_dataframe[max_metric_score_mask]
            best_estimator_name = best_estimator_df.iloc[0]

            best_estimator_object = best_estimators_dictionary[best_estimator_name]

            return best_estimator_df, best_estimator_object

        except Exception as exc:
            return exc

    def retrain_best_estimator_on_whole_train_set(self, X_train, y_train):
        try:
            self.best_estimator.fit(X_train, y_train)

        except Exception as exc:
            return exc
          
def show_mlflow_experiments_info(filter_key, filter_value, model_type=None, get_latest_run=False):
    try:
        query = "tags.{} = '{}'".format(filter_key, filter_value)
        results = mlflow.search_runs(filter_string=query)
        
        if get_latest_run:
            return results.iloc[0]
        
        return results

    except Exception as exc:
        print(exc)
        return exc
      
def sort_by_columns(dataframe, col_to_sort_by=None):
    """Sorts dataframe by columns

    Args:
        col_to_sort_by (array, optional): column/s to sort by. Defaults to None.
    """
    try:
        dataframe = dataframe.sort_values(by=col_to_sort_by)
        
        return dataframe

    except Exception as exc:
        print(exc)
        return exc

# COMMAND ----------

# Load dataset:
data_prep_obj = Preprocessing()
dataset_location = '/dbfs/FileStore/tables/forecaster_data/precipitations_df.csv'

columns_names = ['tmp0', 'tmp1', 'hPa', 'hum', 'pp']
precipitations_df = data_prep_obj.read_csv(dataset_location, cols_to_use=columns_names, sep_val=',')
precipitations_df.head()

# COMMAND ----------

"""
    If there is a row with more than 50% of the attributes with missing values,
    drop the row.
    Otherwise, impute the missing value in that attribute.
"""
row_indexes_to_delete = data_prep_obj.check_row_indexes_to_delete(0.5)
row_indexes_to_delete

# COMMAND ----------

"""
  And now, we check for each attribute, any possible missing values
"""
attributes_missing_counts_dict = {}
for attribute in tqdm(precipitations_df.columns):
    attribute_missing_sub_df = data_prep_obj.check_for_missing_values(attribute)
    if attribute_missing_sub_df is not None:
        attributes_missing_counts_dict[attribute] = len(attribute_missing_sub_df)

attributes_missing_counts_dict

# COMMAND ----------

"""
  Scaling numeric attributes and binarize target
"""
attributes_names = precipitations_df.columns[:-1]
target_name = precipitations_df.columns[-1]

ds_bin_classifier = Binary_classifier(precipitations_df, attributes_names, target_name)
# let's make our target attribute binary:
precipitations_df[target_name] = data_prep_obj.binarize_target_variable(precipitations_df[target_name])
precipitations_df[target_name] = precipitations_df[target_name].apply(lambda x: np.int(x))

ds_preprocessor = Preprocessing(precipitations_df)

# COMMAND ----------

X_train, X_validation, y_train, y_validation = ds_bin_classifier.split_into_train_validation_sets(0.3)

X_train_scaled = ds_preprocessor.standard_scaler_transformer(X_train, X_train.columns, preprocessor_saving_path='preprocessor_scaler.pickle')
X_train = None

# COMMAND ----------

'''
  Training phase.
  We decide a list of desired models to implement in the training process, all
  of them being part of sklearn:
  - Dummy most-frequent classifier
  - Gaussian Naive Bayes classifier
  - Logistic regression classifier
  - Support Vector classifier
'''

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

# COMMAND ----------

#mlflow.sklearn.autolog()

cv_results_df, best_estimators_dict = ds_bin_classifier.select_model_via_grid_search_cv(models_list,
                                                    models_and_params,
                                                    X_train_scaled,
                                                    y_train.values,
                                                    cv_folds=10,
                                                    scoring_metrics=['recall',
                                                                     'f1',
                                                                     'roc_auc'],
                                                    refit_metric='roc_auc')

cv_results_df

# COMMAND ----------

sort_by_columns(show_mlflow_experiments_info('use_case_name', 'rain_forecaster'), col_to_sort_by=['metrics.best_cv_score'])

# COMMAND ----------

