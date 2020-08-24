import logging

logging.basicConfig(filename='binary_classifier.log',level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("Binary_classifier")

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