from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import KFold

from data_loader.webtext_data import *
from quick_analysis.quick_analysis import process_text
# from preprocessing.preprocessing import preprocess_df_text
from util import get_stopwords


def cv_preparation(train_data, test_data=None, k_fold=0):
    if k_fold == 0:
        return train_data, test_data, None
    elif k_fold > 0:
        kf = KFold(n_splits=k_fold)
        return train_data, test_data, [(train, test) for train, test in kf.split(train_data)]
    else:
        raise ValueError("k_fold must be a positive integer")


if __name__ == '__main__':
    df = load_data_chat_logs()
    df_embedding, df_test_embedding = process_text(train_data=df, test_data=None, test_prop=0.2)
    x, y, z = cv_preparation(train_data=df_embedding, test_data=df_test_embedding, k_fold=10)
    print(x.shape)
    print(y.shape)
    for el in z:
        print(el[0].shape, el[1].shape)
