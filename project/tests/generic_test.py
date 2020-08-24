import unittest
import pytest


class CodeTests(unittest.TestCase):
    import pandas as pd

    def test_array_length(self):
        test_array = [1, 2, 3]
        l = len(test_array)
        self.assertEqual(l, 3)

    def test_check_content(self):
        from dataset_EDA.eda import Dataset_eda
        import pandas as pd

        test_df = pd.DataFrame({'A': [1, 4, 4], 'B': [2, 2, 7]})
        ds_eda = Dataset_eda(test_df)
        df_length, head_df, tail_df = ds_eda.check_dataframe_content()
        
        self.assertEqual(df_length, 3)

    def test_target_binarization(self):
        from dataset_preprocessing.preprocessing import Preprocessing
        import pandas as pd

        test_df = pd.DataFrame({'A': [1, 0, 4], 'B': [2, 0, 0.7]})
        ds_preproc = Preprocessing(test_df)
        target_series = ds_preproc.binarize_target_variable(test_df.B)
        target_series = target_series.apply(lambda x: int(x))

        self.assertEqual(target_series.unique().all(), pd.Series([1, 0, 0]).unique().all())