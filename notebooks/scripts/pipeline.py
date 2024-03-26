import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute

#define pipeline for preprocessing as function
def Pipeline(data):
    data = pd.get_dummies(data)
    data['LotFrontage'] = data['LotFrontage'].fillna(0)
    imputer = impute.KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data