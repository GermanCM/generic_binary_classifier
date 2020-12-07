
class Preprocessing():
    """
        This class is ready to receive an existing 'dataframe'
        to implement preprocessing actions
    """
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

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
                logger.exception('raised exception at {}: {}'.format(
                    logger.name + '.' + self.impute_attribute.__name__, exc))

    def binarize_target_variable(self, target_series):
        try:
            binary_target_series_mask = target_series > 0
            target_series[binary_target_series_mask] = 1
            return target_series

        except Exception as exc:
                logger.exception('raised exception at {}: {}'.format(
                                 logger.name + '.' +
                                 self.binarize_target_variable.__name__, exc))

    def standard_scaler_transformer(self, attributes_set, columns_to_scale):
        try:
            from sklearn.preprocessing import StandardScaler
            import pandas as pd
            from sklearn.pipeline import Pipeline
            import pickle 

            scaler = StandardScaler()
            scaler_pipeline = Pipeline([('scaler', scaler)])

            #scaled_df_values = scaler.fit_transform(attributes_set.values)
            fitted_scaler = scaler_pipeline.fit(attributes_set.values)
            pickle.dump(fitted_scaler, open("preprocessor_scaler.pickle", "wb"))

            scaled_df_values = fitted_scaler.transform(attributes_set.values)
            scaled_df = pd.DataFrame(columns=columns_to_scale,
                                     data=scaled_df_values)
            
            return scaled_df

        except Exception as exc:
                logger.exception('raised exception at {}: {}'.format(
                                 logger.name + '.' +
                                 self.standard_scaler_transformer.__name__,
                                 exc))


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
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.split_into_train_validation_sets.__name__, exc))

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
                model = sklearn.base.clone(models_list[i])

                clf = GridSearchCV(model, hyperparams, cv=cv_folds, scoring=scoring_metrics, 
                                   refit=refit_metric, return_train_score=True) 
                clf.fit(X_train, y_train)
              
                cv_model_results=pd.DataFrame(pd.DataFrame(clf.cv_results_))
                cv_results_df = cv_results_df.append(cv_model_results) #, ignore_index=True)

                best_estimators_dict[model_name] = clf.best_estimator_
                self.best_estimator = clf.best_estimator_
                i += 1

            return cv_results_df, best_estimators_dict
        
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.select_model_via_grid_search_cv.__name__, exc))

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
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.choose_best_estimator.__name__, exc))

    def retrain_best_estimator_on_whole_train_set(self, X_train, y_train):
        try:
            self.best_estimator.fit(X_train, y_train)

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.choose_best_estimator.__name__, exc))