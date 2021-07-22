from unittest import TestCase
from src.analysis import analyse
from pandas import util
import numpy as np
import pandas as pd


class TestAnalysis(TestCase):
    def setUp(self) -> None:
        self.test_df = pd.DataFrame(np.random.randint(0,100, size=(100, 3)), columns=['Ab', 'Bc', 'atemp'])
        self.test_df.head()
        self.columns = list(self.test_df.columns)

    def test_analyse(self):
        analyse(self.columns, self.test_df, [], [self.columns[-1]], self.test_df)
        assert('atemp' not in self.columns)


