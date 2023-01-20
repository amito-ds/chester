import pandas as pd

from full_cycle.run_full_cycle import parameter_completer


class A:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
class B:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df


if __name__ == 'main':
    df = pd.DataFrame()
    A_obj = A(df=df)
    B_obj = B()
    parameter_completer(A_obj, B_obj)
