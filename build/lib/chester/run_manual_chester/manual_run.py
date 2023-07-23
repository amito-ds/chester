from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot as plt

from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur, load_data_chat_logs
from chester.run.full_run import run
from chester.run.user_classes import Data, TextSummary, ModelRun

# matplotlib.use('TkAgg')
target_column = 'target'
import numpy as np
# Disable interactive mode
# plt.ioff()


################################################################################################
# df1 = load_data_pirates().assign(target='pirate').sample(300, replace=True)
# df2 = load_data_king_arthur().assign(target='arthur').sample(300, replace=True)
# df3 = load_data_chat_logs().assign(target='chat').sample(300, replace=True)
# df = pd.concat([
#     df1, df2,
#     df3
# ])


#
# df['target'] = df['target'].apply(lambda x: 0 if "pirate" in x else 1)  # can do with or without
################################################################################################


################################################################################################
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)
# dataset.rename(columns={'class': 'target'}, inplace=True)
# df = dataset.sample(frac=1).reset_index(drop=True)
# df['target'] = df['target'].apply(lambda x: 0 if "Iris-setos" in x else 1)  # can do with or without
###############################################################################################

###############################################################################################
# Load the Boston Housing dataset. categorical
# boston = fetch_openml(name='boston', version=1)
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['target'] = boston.target
###############################################################################################

###############################################################################################
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/day.csv")
# df.rename(columns={'cnt': 'target'}, inplace=True)
###############################################################################################


###############################################################################################
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/camps.csv")
# df.rename(columns={'Competitors ': 'target'}, inplace=True)
###############################################################################################


###############################################################################################
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/Reviews.csv")
# df = df[['Text', 'Summary', 'ProductId']]


# df = df[['Text', 'Summary']]
###############################################################################################


###############################################################################################
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/daily_cinema.csv")
# df.rename(columns={'humidity': 'target'}, inplace=True)
##############################################################################################


###############################################################################################
def load_marvel_goodbad():
    heroes_information = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/heroes_information.csv")
    super_hero_powers = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/super_hero_powers.csv")
    df1 = pd.merge(heroes_information, super_hero_powers,
                   left_on='name', right_on='hero_names', how='left')

    df1.rename(columns={'Alignment': 'target'}, inplace=True)
    df1.drop(columns=['Unnamed: 0', 'name', 'hero_names'],
             inplace=True)

    # define a translation table to remove the specified characters

    def translate(str1):
        str1 = str1.lower()
        str1 = str1.replace('/', '')
        str1 = str1.replace('.', '')
        str1 = str1.replace('-', '')
        str1 = str1.replace('*', '')
        return str1

    # map the columns to their cleaned names using the translation table
    column_mapping = {col: translate(col) for col in df1.columns}
    df1 = df1.rename(columns=column_mapping)
    df1 = df1[df1['target'].isin(["good", "bad"])]

    return df1


# df = load_marvel_goodbad()
##############################################################################################

##############################################################################################
def load_maverl_dc():
    marvel = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/marvel-wikia-data.csv")
    dc = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/dc-wikia-data.csv")
    data1 = pd.concat([marvel, dc])
    data1.rename(columns={'ALIGN': 'target'}, inplace=True)
    data1.drop(columns=['urlslug', 'name'], inplace=True)
    return data1


# df = load_maverl_dc()
# print(df[0:10])
##############################################################################################

##############################################################################################
def load_autizm():
    data_aut = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/autism_prediction.csv")
    data_aut.rename(columns={'Class/ASD': 'target'}, inplace=True)
    return data_aut


# df = load_autizm()
##############################################################################################


###############################################################################################
# df = pd.read_csv("chester/model_training/models/chester_models/day.csv")
# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/time_series_kaggle.csv")
# df = df[['date', 'store_nbr', 'family', 'sales']]

# top_stores = df.groupby('store_nbr').size().sort_values(ascending=False)[:5].index
# df = df[df['store_nbr'].isin(top_stores)]
# df.rename(columns={'sales': 'target'}, inplace=True)
###############################################################################################


################################################################################################
# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# df = pd.DataFrame(X)
# df.rename(columns={col: "feat_" + str(col) for col in df.columns}, inplace=True)
# df['target'] = digits.target
# df['target'] = "c_ " + df['target'].astype(str)
################################################################################################


################################################################################################
# categorical features
# import seaborn as sns
# df = sns.load_dataset("tips")
# df.rename(columns={'tip': target_column}, inplace=True)
################################################################################################

################################################################################################
# categorical
# import seaborn as sns
# df = sns.load_dataset("titanic")
# df.rename(columns={'survived': target_column}, inplace=True)
# df.drop(columns=['alive'], inplace=True)
###############################################################################################

###############################################################################################


################################################################################################
# df = pd.read_csv("chester/run_manual_chester/HR_Analytics.csv.csv")
# df.rename(columns={'Attrition': 'target'}, inplace=True)
###############################################################################################

###############################################################################################
# df = pd.read_csv("chester/run_manual_chester/flipkart_product.csv", encoding="ISO-8859-1")
# df = df[['Review', 'Summary', 'Rate']]
# df['Rate'] = "category: " + df['Rate'].astype(str)
# df.rename(columns={'Rate': 'target'}, inplace=True)
# df.dropna(subset=['target'], inplace=True)
# df = df.sample(5000)
###############################################################################################


###############################################################################################
# df1 = pd.read_csv("chester/run_manual_chester/Bank_Personal_Loan_Modelling.csv")
# df = df1.copy(deep=False)
# df.rename(columns={'Personal Loan': 'target'}, inplace=True)
#
# feature_types = {'numeric':
#                      ['ID', 'Age', 'Experience', 'Income', 'Family',
#                       'CCAvg', 'Education', 'Mortgage', 'Securities Account',
#                       'CD Account', 'Online', 'CreditCard'], 'boolean': [],
#                  'text': [], 'categorical': ['Education'], 'time': []}
###############################################################################################


###############################################################################################
# df1 = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/run_manual_chester/Hotel Reservations.csv")
# df = df1.copy(deep=False)
# df.rename(columns={'booking_status': 'target'}, inplace=True)
# feature_types = {'numeric': ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
#                              'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month',
#                              'arrival_date', 'repeated_guest', 'no_of_previous_cancellations',
#                              'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
#                              'no_of_special_requests'],
#                  'boolean': [],
#                  'text': [],
#                  'categorical': ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], 'time': []}
###############################################################################################
# df1 = pd.read_csv("chester/run_manual_chester/loans_1.csv")
# df = df1.copy(deep=False)
# df.rename(columns={'Loan_Status': 'target'}, inplace=True)
# df.dropna(subset=['target'], inplace=True)
###############################################################################################

###############################################################################################


def generate_data(n_features, n_rows, target_type='binary'):
    if target_type == 'binary':
        # Create binary target column
        target = np.random.choice(['yes', 'no'], size=n_rows)
    elif target_type == 'multiclass':
        # Create multiclass target column
        target = np.random.choice(['class_1', 'class_2', 'class_3', 'class_4',
                                   'class_5', 'class_6', 'class_7', 'class_8',
                                   'class_9', 'class_10', 'class_11', 'class_12'], size=n_rows)
    else:
        raise ValueError("Invalid target_type. Must be either 'binary' or 'multiclass'.")

    # Create feature categorical columns
    features = {}
    for i in range(n_features):
        feature = np.random.choice(['A', 'B', 'C', 'D'], size=n_rows)
        features[f'feature_{i}'] = feature

    # Create pandas DataFrame
    df = pd.DataFrame(features)
    df['target'] = target

    return df


# df = generate_data(1, 100, target_type='binary')
# df = generate_data(30, 100, target_type='multiclass')
###############################################################################################


def load_ex2():
    newsgroups_train = fetch_20newsgroups(subset='train')
    df = pd.DataFrame(newsgroups_train.data, columns=['text'])
    y = newsgroups_train.target
    df['target'] = y
    category_counts = Counter(y)
    top_3_categories = category_counts.most_common(4)
    top_3_categories = [cat for cat, count in top_3_categories]
    df = df[df.target.isin(top_3_categories)].sample(1500)
    df['target'] = "c_ " + df['target'].astype(str)
    return df


def load_ex3():
    train_data, _ = ml_datasets.dbpedia()
    train_data = pd.DataFrame(train_data, columns=["text", "target"])
    train_data['target'] = "c_: " + train_data['target'].astype(str)
    return train_data


def load_ex4():
    train_data, _ = ml_datasets.cmu()
    train_data = pd.DataFrame(train_data, columns=["text", "target"])
    train_data['target'] = "category: " + train_data['target'].astype(str)
    train_data = train_data[train_data['target'].isin(train_data['target'].value_counts()[:5].index)]
    return train_data


def load_ex5():
    train_data, _ = ml_datasets.quora_questions()
    train_data = [(a, b, c) for ((a, b), c) in train_data]
    train_data = pd.DataFrame(train_data, columns=["text1", "text2", "target"])
    return train_data


# load data
# df = load_ex2()
# df = load_ex3().sample(1000)
# df = load_ex4().sample(1000)
# df = load_ex5().sample(900)


def load_yaho(tickers=None, start_date='2010-01-01', end_date='2023-02-15'):
    # Download stock data for all tickers and concatenate them into a single dataframe
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOG']
    df_list = []
    for ticker in tickers:
        ticker_df = yf.download(ticker, start=start_date, end=end_date)
        ticker_df["id"] = ticker
        df_list.append(ticker_df[['Close', 'id']])

    # Combine all dataframes into one and reset the index
    df = pd.concat(df_list).reset_index()

    # Rename the columns to match the desired output
    df = df.rename(columns={'Date': 'date', 'Close': 'value'})

    # Rearrange the columns to match the desired output
    df = df[['id', 'date', 'value']]
    df.rename(columns={'value': 'target'}, inplace=True)

    return df


# df = load_yaho()
# print("df shape", df.shape)
# print("df cols", df.columns)

import pandas as pd
from sklearn.datasets import fetch_openml


def load_dataset(name):
    print("DB name:", name)
    data = fetch_openml(name=name, as_frame=True)
    df = data.data
    df['target'] = data.target
    # df['target'] = df['target'].apply(lambda x: 'class_' + str(x))
    df = df.dropna(axis=1, how='all')  # remove nas
    return df


# df = load_dataset('arrhythmia')
# df['target'] = df['target'].cat.codes

# df = load_dataset('labor')

# df = load_dataset('arrhythmia')
# df['target'] = df['target'].cat.codes

# df = load_dataset('letter')
# df['target'] = df['target'].cat.codes

# df = load_dataset('audiology')

# df = load_dataset('autos')

# df = load_dataset('lymph')
# df = load_dataset('balance-scale')
# df = load_dataset('mfeat-factors')
# df = load_dataset('breast-cancer')
# df = load_dataset('mfeat-fourier')
# df = load_dataset('breast-w')
# df = load_dataset('mfeat-karhunen')
# df = load_dataset('mfeat-morphological')
# df = load_dataset('mfeat-pixel')
# df = load_dataset('mfeat-zernike')
# df = load_dataset('cmc')
# df = load_dataset('mushroom')
# df = load_dataset('colic')
# df = load_dataset('optdigits')
# df = load_dataset('credit-approval')
# df = load_dataset('page-blocks')
# df = load_dataset('credit-g')
# df = load_dataset('pendigits')
# df = load_dataset('postoperative-patient-data')
# df = load_dataset('dermatology')
# df = load_dataset('segment')
# df = load_dataset('diabetes')
# df = load_dataset('sick')
# df = load_dataset('ecoli')
# df = load_dataset('sonar')
# df = load_dataset('glass')
# df = load_dataset('soybean')
# df = load_dataset('haberman')
# df = load_dataset('spambase')
# df = load_dataset('splice')
# df = load_dataset('tae')
# df = load_dataset('heart-c')
# df = load_dataset('tic-tac-toe')
# df = load_dataset('heart-h')
# df = load_dataset('trains')
# df = load_dataset('heart-statlog')
# df = load_dataset('vehicle')
df = load_dataset('hepatitis')
# df = load_dataset('vote')
# df = load_dataset('hypothyroid')


# df = load_dataset('ionosphere')
# df = load_dataset('waveform-5000')
# df = load_dataset('iris')
# df = load_dataset('zoo')
# df = load_dataset('BNG(anneal,nominal,1000000)')
# df = load_dataset('BNG(anneal.ORIG,nominal,1000000)')
# df = load_dataset('BNG(kr-vs-kp)')
# df = load_dataset('BNG(colic,nominal,1000000)')
# df = load_dataset('BNG(optdigits)')
# df = load_dataset('BNG(credit-a,nominal,1000000)')
# df = load_dataset('BNG(page-blocks,nominal,295245)')
# df = load_dataset('BNG(credit-g,nominal,1000000)')
# df = load_dataset('BNG(cylinder-bands,nominal,1000000)')
# df = load_dataset('BNG(segment)')
# df = load_dataset('BNG(glass,nominal,137781)')
# df = load_dataset('BNG(soybean)')
# df = load_dataset('CovPokElec')
# df = load_dataset('covertype')
# df = load_dataset('electricity')

# df = load_dataset('pokerhand')
# df = load_dataset('SEA(50)')
# df = load_dataset('lung-cancer')
# df = load_dataset('molecular-biology_promoters')
# df = load_dataset('primary-tumor')
# df = load_dataset('shuttle-landing-control')
# df = load_dataset('adult')
# df = load_dataset('covertype')
# df = load_dataset('yeast')
# df = load_dataset('satimage')
# df = load_dataset('abalone')
# df = load_dataset('kropt')
# df = load_dataset('yokohoma2')
# df = load_dataset('cpu_small')

# top_10 = df['ProductId'].value_counts().nlargest(10).index
# filtered_df = df[df['ProductId'].isin(top_10)]


# df = pd.read_csv("/Users/amitosi/PycharmProjects/databot_aws/example_app/static/iris_data.csv")
# df['target'] = df.apply(lambda x: str(x['target']) + "class", axis=1)

chester_collector = run(Data(df=df, target_column='target'),
                        # is_feature_stats=True,
                        # is_pre_model=True,
                        # text_summary=TextSummary(summary_num_sentences=3, max_terms=5),
                        # feature_types={'numeric': [], 'boolean': [], 'text': ['Text', 'Summary'],
                        #                'categorical': [], 'time': [], 'id': ['ProductId']}
                        # text_summary=TextSummary(summary_num_sentences=3, max_terms=20),
                        model_run=ModelRun(n_models=2),
                        )
# madcat_collector = run_madcat(Data(df=df, target_column='target'),
#                               is_feature_stats=True,
#                               # text_feature_extraction=TextFeatureExtraction(corex_dim=2, anchor_words=anchors),
#                               # time_series_handler=TimeSeriesHandler(id_cols=["id"]),
#                               is_pre_model=False,
#                               is_model_training=False,
#                               model_run=ModelRun(n_models=2),
#                               is_post_model=False, is_model_weaknesses=False,
#                               plot=False,
#                               # feature_types={'numeric': [], 'boolean': [], 'text': ['Campaign Name'],
#                               #                'categorical': ['Campaign Name'], 'time': ['Date'], 'id': []}
#                               )

# output_collector = run_madcat(
#     Data(df=df, target_column='target'),
#     model_run=ModelRun(n_models=10),
#     is_feature_stats=True,
#     is_pre_model=True,
#     # feature_types=feature_types,
#     is_model_training=True, is_post_model=True, is_model_weaknesses=True
# )
