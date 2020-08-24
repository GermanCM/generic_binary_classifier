#%%
import logging

logging.basicConfig(filename='classifier.log',level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("Dataset_eda")

#%%
class Dataset_eda():
    """
        This class is ready to receive an existing 'dataframe' to implement several EDA steps
    """
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def check_dataframe_content(self):
        try:
            df_length = len(self.dataframe)
            head_df = self.dataframe.head(5)
            tail_df = self.dataframe.tail(5)

            return df_length, head_df, tail_df

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.check_dataframe_content.__name__, exc))

    def profile_dataframe(self, output_file_location_name=None):
        """
        'profile_dataframe' function lets you create a profile report for the EDA phase of your data science project,
        displaying some statistical properties of the dataframe attributes together with visualizations
        """
        try:
            import pandas_profiling as pp

            profile = pp.ProfileReport(self.dataframe)
            profile.to_file(outputfile=output_file_location_name)
            return
    
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.profile_dataframe.__name__, exc))

    '''
    'get_rejected_attributes' returns the attributes wich would be rejected based on a defined correlation threshold,
    i.e., attributes having a correlation > threshold could be rejected  
    '''
    def get_rejected_attributes(self, correlation_threshold=0.9):
        try:
            import pandas_profiling

            array_df_profile = pandas_profiling.ProfileReport(self.dataframe)
            array_df_rejected = array_df_profile.get_rejected_variables(correlation_threshold)
            
            return array_df_rejected

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.get_rejected_attributes.__name__, exc))

    def check_values_range(self, attribute, expected_range_min, expected_range_max):
        try:
            below_range_mask = self.dataframe[attribute].values < expected_range_min
            above_range_mask = self.dataframe[attribute].values > expected_range_max
            below_range_samples = self.dataframe[below_range_mask]
            above_range_samples = self.dataframe[above_range_mask]

            return below_range_samples, above_range_samples
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.check_values_range.__name__, exc))

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
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.get_outliers_tukey_test.__name__, exc))

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
            nan_values_mask = np.isnan(dataframe[attribute])==True
        
            return dataframe[nan_values_mask] 

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.check_for_missing_values.__name__, exc))

    def check_row_indexes_to_delete(self, missing_threshold=0.5):
        try:
            not_missing_counts = self.dataframe.count(axis='columns')
            no_missing_rates = not_missing_counts/len(self.dataframe.columns)
            missing_rate_over_threshold_mask = no_missing_rates < missing_threshold 

            row_indexes_to_delete = no_missing_rates[missing_rate_over_threshold_mask].index
            return row_indexes_to_delete

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.check_for_missing_values_rate_per_attribute.__name__, exc))
            
