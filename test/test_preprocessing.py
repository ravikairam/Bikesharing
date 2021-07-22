from unittest import TestCase
from src.preprocessing import split_and_select_features
import numpy as np
import pandas as pd


class PreprocessingTest(TestCase):
    def setUp(self) -> None:
        self.test_df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=['Ab', 'Bc', 'atemp'])
        self.test_df.head()
        self.columns = list(self.test_df.columns)

    def test_split_and_select_features(self):
        features, number_features, target, test, train, val = split_and_select_features(self.test_df)
        assert(len(train) == 60)
        assert(len(test) == 20)
        assert(len(val) == 20)
        assert(target == ['cnt'])
        assert('temp' in features)
        assert('atemp' in features)

