import pandas as pd
import pickle
import sys

#def load_model()

def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: dataset
    Returns
    -------
    pd.DataFrame
    """
    try:
        #preprocess the input
        scaler_preprocessor = pickle.load(open(sys.path[0]+"/preprocessor_artifacts/preprocessor_scaler.pkl", "rb"))
        input_data = pd.DataFrame(data)
        
        scaled_input_sample = scaler_preprocessor.transform(input_data[input_data.columns[:-1]])

        return scaled_input_sample

    except Exception as exc:
        print('at transform: ',exc)
        return exc

