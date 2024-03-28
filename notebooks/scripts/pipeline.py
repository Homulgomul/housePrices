import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute

#define pipeline for preprocessing as function
def Pipeline(data):
    data = pd.get_dummies(data)
    data = data.drop("Id", axis=1)
    data['LotFrontage'] = data['LotFrontage'].fillna(0)
    for col in data.columns:
        if (data[col].dtype == 'int64' or data[col].dtype == 'float64'):
            data[col] = data[col].clip(lower=data[col].quantile(0.001), upper=data[col].quantile(0.999))
    imputer = impute.KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data