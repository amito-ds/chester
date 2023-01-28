import unittest
import pandas as pd

from chester.zero_break.problem_specification import DataInfo


class TestDataSpec(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.target = 'a'

    def test_has_target_true(self):
        spec = DataInfo(self.data, target=self.target)
        self.assertTrue(spec.has_target())

    def test_has_target_false(self):
        spec = DataInfo(self.data)
        self.assertFalse(spec.has_target())

    ## test problem type
    def test_problem_type(self):
        # Test case 1: No target variable
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}))
        assert data_info.problem_type() == "No target variable"

        # Test case 1.5: No target variable
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'target': [4, 5, 6]}))
        assert data_info.problem_type() == "No target variable"

        # Test case 2: Binary regression
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': [0, 1, 1]}), target='target')
        assert data_info.problem_type() == "Binary regression"
        #
        # # Test case 3: Regression
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': [1.2, 3.4, 5.6]}), 'target')
        assert data_info.problem_type() == "Regression"
        #
        # # Test case 4: Binary classification
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': ['a', 'b', 'b']}), 'target')
        assert data_info.problem_type() == "Binary classification"

        # Test case 5: Multiclass classification
        data_info = DataInfo(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': ['a', 'b', 'c']}), 'target')
        assert data_info.problem_type() == "Multiclass classification"
        #
        # # Test case 6: Binary classification with single column target
        data_info = DataInfo(pd.DataFrame({'target': [0, 1, 1]}), 'target')
        assert data_info.problem_type() == "Binary regression"
        #
        # # Test case 7: Multiclass classification with single column target
        data_info = DataInfo(pd.DataFrame({'target': ['a', 'b', 'c']}), 'target')
        assert data_info.problem_type() == "Multiclass classification"
        #
        # # Test case 8: Numeric regression with single column target
        data_info = DataInfo(pd.DataFrame({'target': [1.2, 3.4, 5.6]}), 'target')
        assert data_info.problem_type() == "Regression"
        #
        # # Test case 9: Categorical regression with single column target
        data_info = DataInfo(pd.DataFrame({'target': ['a', 'b', 'b']}), 'target')
        assert data_info.problem_type() == "Binary classification"


if __name__ == '__main__':
    unittest.main()
