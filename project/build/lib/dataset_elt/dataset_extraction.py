'''
This module contains methods to extract and load the necessary datasets of the use case.
It can do it as: 
    * a usual Pandas read_csv action
    * read it from a Onesait ontology 
'''
import logging

logging.basicConfig(filename='classifier.log',level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("Dataset_preprocessing")

class Dataset_extraction():
    """
    - return_query_result: makes a request to the 'entity_name' ontology <p>
    - build_data_frame_struct: builds up the container (dataframe) structure <p>
    - fill_dataframe: fills the built structure in the former step with the data contined in the 'query_result' <p>
    - build_dataframe_from_query_base: builds a dataframe in chuncks using the 'fill_dataframe' function 
    - load_dataset: makes use of the above methods to load from a csv file or an ontology

    input params: 
        for Onesait mode:
        - iot_client_host: server URL where the ontology is stored
        - iot_client_name: client name whith granted access  
        - iot_client_token: valid token
        for csv mode:
        - dataset_location: complete file location in the style of: "C:\\Users\\local_user_name\\Documents\\Git_repositories_folder\\repository_folder_name\\datasets\\file_name.csv"
    """
    def __init__(self, dataset_location=None, iot_client_host=None, iot_client_name=None, iot_client_token=None):
        self.dataset_location = dataset_location
        self.iot_client_name = iot_client_name
        self.iot_client_host = iot_client_host 
        self.iot_client_token = iot_client_token
    
    def return_query_result(self, client_object, _ontology, desired_query):
        try:
                client_object.connect()
                _ok, _res_from_query = client_object.query(query=desired_query, ontology=_ontology , query_type="SQL")
                client_object.leave()

                return _res_from_query
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.return_query_result.__name__, exc))
    
    def build_data_frame_struct(self, columns_series, column_name_index=None):
        import pandas as pd
        try:
            df_struct = pd.DataFrame(columns=columns_series)
            return df_struct

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.build_data_frame_struct.__name__, exc))
                
    def fill_dataframe(self, final_dataframe_struct, query_result, selected_fields, ontology_first_level_name=None,
                    ontology_items_field_name='items', ontology_timestamp_field_name='Timestamp'):
        import pandas as pd
        from tqdm import tqdm
        try:
            final_dataframe = pd.DataFrame(columns=selected_fields)
            
            for element in tqdm(query_result):
                    first_level_elements = pd.Series(element[ontology_first_level_name])
                    items_elements = pd.Series(element[ontology_first_level_name][ontology_items_field_name])
                    for selected_field in selected_fields:
                        if selected_field in items_elements.index:
                            final_dataframe.loc[element[ontology_first_level_name][ontology_timestamp_field_name], selected_field] = items_elements[selected_field]
                        elif selected_field in first_level_elements.index:
                            final_dataframe.loc[element[ontology_first_level_name][ontology_timestamp_field_name], selected_field] = first_level_elements[selected_field]
                        else:
                            continue

            return final_dataframe    
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.fill_dataframe.__name__, exc))

    def build_dataframe_from_query_base(self, client_object, ontology, columns_names, desired_base_query=None, column_name_index=None, limit_rows_number=10000,
                            query_rows_number=None, initial_date_from=None, ontology_first_level_name=None, ontology_items_field_name='items', ontology_timestamp_field_name='Timestamp'): 
        import pandas as pd
        import time
        from tqdm import tqdm    

        df_query_result = pd.DataFrame(columns=columns_names)
        try:
            if query_rows_number is None:
                client_object.connect()
                _ok, _res_from_query = client_object.query(query=desired_base_query, ontology=ontology , query_type="SQL")
                client_object.leave()
                columns_names.append(column_name_index)
                res_from_query_df_struct = self.build_data_frame_struct(columns_names)
                df_query_result = self.fill_dataframe(final_dataframe_struct=res_from_query_df_struct, query_result=_res_from_query, 
                                                      selected_fields=columns_names, ontology_first_level_name=ontology_first_level_name,
                                                      ontology_items_field_name=ontology_items_field_name, 
                                                      ontology_timestamp_field_name=ontology_timestamp_field_name)

                return df_query_result
            else:
                if column_name_index is not None:
                    columns_names.append(column_name_index)
                date_from = initial_date_from
                res_from_query_df_struct = self.build_data_frame_struct(columns_names)

                for chunk in tqdm(range(0, query_rows_number, limit_rows_number)): 
                    client_object.connect()
                    if initial_date_from is not None: 
                        desired_query_with_chunk_limit = desired_base_query + ' WHERE ' + _ontology + '.' + ontology_timestamp_field_name + ' >= TIMESTAMP("' + date_from + '") LIMIT ' + str(limit_rows_number)
                    else:
                        desired_query_with_chunk_limit = desired_base_query + ' LIMIT ' + str(limit_rows_number)

                    _ok, _res_from_query = client_object.query(query=desired_query_with_chunk_limit , ontology=ontology , query_type="SQL")
                    client_object.leave()
                    if not _ok: 
                        continue

                    df_query_result_sub = self.fill_dataframe(res_from_query_df_struct, _res_from_query, columns_names, ontology_first_level_name,
                                                              ontology_items_field_name, ontology_timestamp_field_name)
                    
                    df_query_result = df_query_result.append(df_query_result_sub, sort=False)
                    date_from = _res_from_query[-1][ontology_first_level_name][ontology_timestamp_field_name]
                return df_query_result
        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.build_dataframe_from_query_base.__name__, exc))
            return df_query_result

    def load_dataset(self, csv_mode=False, separator=';', ontology_name=None, columns_names=None, date_from=None, 
                     date_to=None, desired_query_base=None, limit_rows_number=5000, ontology_first_level_name=None, 
                     ontology_items_field_name='items'):
        try:
            if csv_mode:
                import pandas as pd 
                
                dataset_location_route = self.dataset_location 
                extracted_dataset = pd.read_csv(dataset_location_route, sep=separator)

                return extracted_dataset
            else:
                from onesaitplatform.iotbroker import DigitalClient #IotBrokerClient
                
                iot_broker_client = IotBrokerClient(host=self.iot_client_host,
                            iot_client=self.iot_client_name, iot_client_token=self.iot_client_token)
                iot_broker_client.protocol = "http"
                iot_broker_client.avoid_ssl_certificate = True

                self.iot_broker_client = iot_broker_client

                if (date_from is not None) and (date_to is not None):
                    desired_query = 'SELECT COUNT(*) from ' + ontology_name + ' as c where ' + ontology_name + '.Timestamp >= TIMESTAMP("' + date_from + '") AND ' + ontology_name + '.Timestamp < TIMESTAMP("' + date_to + '")'
                else:
                    desired_query = 'SELECT COUNT(*) from ' + ontology_name + ' as c'
                
                rows_counter_query = self.return_query_result(iot_broker_client, ontology_name, desired_query)
                if len(rows_counter_query)==0:
                    raise Exception("No rows were found fulfilling the requested query")

                number_of_rows = rows_counter_query[0]['value']
                if desired_query_base is None: 
                    desired_query_base = 'select * from ' + ontology_name + ' as c' #to adapt with column names
                
                _res_df = self.build_dataframe_from_query_base(client_object=iot_broker_client, ontology=ontology_name, 
                    columns_names=columns_names, desired_base_query=desired_query_base, limit_rows_number=limit_rows_number,
                    ontology_items_field_name=ontology_items_field_name, ontology_first_level_name = ontology_first_level_name,
                    query_rows_number=number_of_rows)

                return _res_df

        except Exception as exc:
            logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.load_dataset.__name__, exc))
#%%
