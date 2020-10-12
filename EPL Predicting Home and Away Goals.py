import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set option to display all the rows and columns in the dataset. If there are more rows, adjust number accordingly.
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Files
data_train = pd.read_excel(r"C:\Users\harsh\Documents\My Dream\Desktop\Betting\Updated.xlsx", sheet_name='train')
data_test = pd.read_csv(r"C:\Users\harsh\Documents\My Dream\Desktop\Betting\test_EPL.csv")


def DataDesc(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2)

    return summary


# Keeping columns that we want
dtr = data_train
dte = data_test.drop(columns=['Location', 'Date', 'Round Number', 'Id'])
dtr = dtr.loc[:, dtr.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5'])]

# Filling NA values, converting columns and column types.. 1.5 works better with actual scores than 1.9
dte = dte.fillna(1.5)
dtr = dtr.fillna(1.5)

rcolumndict = {'FTHG': int,
               'HomeTeam': str,
               'AwayTeam': str,
               'FTAG': int,
               'B365H': float,
               'B365D': float,
               'B365A': float,
               'B365>2.5': float,
               'B365<2.5': float
               }

dtr = dtr.astype(rcolumndict)

dtr_h = dtr.dropdtr = dtr.drop(['FTAG', 'FTR'], axis=1)

dtr = dtr.drop(['FTHG', 'FTR'], axis=1)
dte = dte.drop(['FTHG', 'FTAG', 'FTR'], axis=1)

print('DTE')
print(DataDesc(dte))
print('DTR')
print(DataDesc(dtr))
print('DTR_H')
print(DataDesc(dtr_h))

d_test = dte
d_train = dtr
d_train_h = dtr_h

from sklearn import preprocessing


def encode_features(df_train, df_test, df_train_h):
    features = ['HomeTeam', 'AwayTeam']
    df_combined = pd.concat([df_train[features], df_test[features], df_train_h[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
        df_train_h[feature] = le.transform(df_train_h[feature])
    return df_train, df_test, df_train_h


# dtr_a, dte = encode_features(dtr_a, dte)

dtr, dte, dtr_h = encode_features(dtr, dte, dtr_h)

X = dtr.drop(['FTAG'], axis=1)
y = dtr['FTAG']

X_h = dtr_h.drop(['FTHG'], axis=1)
y_h = dtr_h['FTHG']

train_X_h, val_X_h, train_y_y, val_y_y = train_test_split(X_h, y_h, random_state=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model_on_full_data = RandomForestRegressor()
rf_model_on_full_data_h = RandomForestRegressor()

rf_model_on_full_data.fit(X, y)

rf_model_on_full_data_h.fit(X_h, y_h)

test_preds_h = rf_model_on_full_data_h.predict(dte)

test_preds = rf_model_on_full_data.predict(dte)

output = pd.DataFrame({'Id': dte.index,
                       'FTHG': test_preds_h,
                       'FTAG': test_preds})
output.to_csv(r"C:\Users\harsh\Documents\My Dream\Desktop\Machine Learning\Attempt 3\output.csv", index=False)
print(output)
