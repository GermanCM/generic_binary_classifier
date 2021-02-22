# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import json
import pickle
import numpy as np 
import pandas as pd 

def make_predictions(input_data):
    """
    Parameters
    ----------
    data : 
        Or if using JSON as input:
        [{"Feature1":numeric_value,"Feature2":"string"}]
 
    Returns
    -------
    Response schema
 
    """
    try:
        #preprocess the input
        scaler_preprocessor = pickle.load(open(sys.path[0] + "/preprocessor_scaler.pickle", "rb"))
        input_data = pd.DataFrame(input_data)
        
        scaled_input_sample = scaler_preprocessor.transform(input_data.values) 
        # trained model load
        selected_model_loaded = pickle.load(open(sys.path[0] + "/selected_model.pickle", "rb"))
        predictions = selected_model_loaded.predict_proba(scaled_input_sample)
        
        input_data = None
        # Return a Python dict following the schema in the documentation
        return predictions #.json()

    except Exception as exc:
        print('at make_predictions: ',exc)
        return 1

def main(input_data):
    """
    Return an exit code on script completion or error. Codes > 0 are errors to the shell.
    
    """
    predictions = None
    try:
        print(dict(input_data))
        input_data = pd.DataFrame(input_data)
        predictions = make_predictions(input_data)
        #print(json.dumps(predictions, indent=4))
        print(predictions)
        return 0
        
    except Exception as exc:
        print('at main: ',exc)
 

if __name__ == "__main__":
    '''input_data = pd.DataFrame([{'tmp0': 16.77, 'tmp1': 16.77, 'hPa': 1053.0, 'hum': 0.61}, 
                  {'tmp0': 12.77, 'tmp1': 16.77, 'hPa': 1053.0, 'hum': 0.61}])'''
    input_data = sys.argv[1:]
    sys.exit(main(input_data))
 
