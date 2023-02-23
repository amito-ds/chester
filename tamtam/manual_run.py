import tamtam
import pandas as pd

from tamtam.ab_info.ab_class import ABInfo
from tamtam.run import run
from tamtam.user_class.user_class import ABData, TestInfo

# read
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/tamtam/data/cookie_cats.csv")
# df["side"] = df["version"].apply(lambda x: "B" if x == "gate_30" else "A")
# df.drop(columns=["version"], inplace=True)
# ab_data = ABData(df)
# test_info = TestInfo(side_col="side", metrics=['retention_1', 'retention_7'], id_cols=['userid'],
#                      feature_cols=['sum_gamerounds'])

# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/tamtam/data/ab_data.csv")
# df["side"] = df["group"].apply(lambda x: "B" if x == "treatment" else "A")
# df.drop(columns=["group"], inplace=True)
#
# # Prepare
# ab_data = ABData(df.sample(5000))
# test_info = TestInfo(side_col="side", metrics=['converted'], id_cols=['user_id'], feature_cols=['landing_page'],
#                      date_col="timestamp")
#
# # run
# run(ab_data=ab_data, test_info=test_info)


df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/tamtam/data/AB_Test_Results.csv")
df["side"] = df["VARIANT_NAME"].apply(lambda x: "B" if x == "variant" else "A")
df.drop(columns=["VARIANT_NAME"], inplace=True)

# Prepare
ab_data = ABData(df.sample(5000))
test_info = TestInfo(side_col="side", metrics=['REVENUE'], id_cols=['USER_ID'])

# run
run(ab_data=ab_data, test_info=test_info)
