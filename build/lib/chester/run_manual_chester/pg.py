import pandas as pd
from chester.run import full_run as fr
from chester.run import user_classes as uc
from matplotlib import pyplot as plt

plt.ioff()
df = pd.read_csv("/Users/amitosi/PycharmProjects/databot_aws/example_app/static/iris_data.csv")
df['target'] = df.apply(lambda x: str(x['target']) + "class", axis=1)
fr.run(uc.Data(df=df, target_column='target'), model_run=uc.ModelRun(n_models=2))
