import pandas as pd
import pickle
import sys

def transform(input_data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """
    try:
        #preprocess the input
        scaler_preprocessor = pickle.load(open(sys.path[0] + "/preprocessor_scaler.pickle", "rb"))
        input_data = pd.DataFrame(input_data)
        
        scaled_input_sample = scaler_preprocessor.transform(input_data.values) 
        
        return scaled_input_sample

    except Exception as exc:
        print('at transform: ',exc)
        return 1
