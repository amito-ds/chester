import tamtam
import pandas as pd

from tamtam.ab_info.ab_class import ABInfo
from tamtam.run import run
from tamtam.user_class.user_class import ABData, TestInfo

# read
df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/tamtam/data/cookie_cats.csv")
df["side"] = df["version"].apply(lambda x: "B" if x == "gate_30" else "A")
df.drop(columns=["version"], inplace=True)

# Prepare
ab_data = ABData(df.sort_values(by='userid')[0:5000])
test_info = TestInfo(side_col="side", metrics=['retention_1', 'retention_7'], id_cols=['userid'],
                     feature_cols=['sum_gamerounds'])

# run
run(ab_data=ab_data, test_info=test_info)
