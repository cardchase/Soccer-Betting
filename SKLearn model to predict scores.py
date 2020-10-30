print("Importing libraries")
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
start = datetime.now()
print("Importing libraries completed")
# Set option to display all the rows and columns in the dataset. If there are more rows, adjust number accordingly.
print("Set Option")
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print("Define Data Describe")


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


print("Getting files")
# Files
data_train = pd.read_csv(r"C:\Users\harsh\Documents\My Dream\sportsintel.shop\Version 2\Example Files\dataset_updated_ppl.csv")
data_test = pd.read_csv(
    r"C:\Users\harsh\Documents\My Dream\sportsintel.shop\Version 2\Example Files\test_PPL - Copy.csv")

print("Reading files complete")

print("Keeping columns and filling all na values")
dtr = data_train
dte = data_test
dtr = dtr.loc[:, dtr.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5'])]

# Filling NA values, converting columns and column types.. 1.5 works better with actual scores than 1.9
dte = dte.fillna(1.5)
dtr = dtr.fillna(1.5)
print("Defining column types")
rcolumndict = {
    'HomeTeam': str,
    'AwayTeam': str,
    'FTHG': int,
    'FTAG': int,
    'B365H': float,
    'B365D': float,
    'B365A': float,
    'B365>2.5': float,
    'B365<2.5': float
}
print("Applying column types for Training data")
dtr = dtr.astype(rcolumndict)

print("Using columns for training and testing")
dtr_h = dtr.dropdtr = dtr.drop(['FTAG'], axis=1)
dtr_a = dtr.drop(['FTHG'], axis=1)
dte = dte.drop(['FTHG', 'FTAG'], axis=1)

print('DTE')
print(dte.head(2))
print(dte.count())
print('DTR_A')
print(dtr_a.head(2))
print(dtr_a.count())
print('DTR_H')
print(dtr_h.head(2))
print(dtr_h.count())

print("Now the magic starts.. MACHINE LEARNING BITCH!")
d_test = dte
d_train_a = dtr_a
d_train_h = dtr_h

print("SK Learn")
print("Pre-processing")
from sklearn import preprocessing

print("Defining feature columns 'text = team names'")
def encode_features(df_train_a, df_test, df_train_h):
    features = ['HomeTeam', 'AwayTeam']
    df_combined = pd.concat([df_train_a[features], df_test[features], df_train_h[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train_a[feature] = le.transform(df_train_a[feature])
        df_test[feature] = le.transform(df_test[feature])
        df_train_h[feature] = le.transform(df_train_h[feature])
    return df_train_a, df_test, df_train_h

print("Encoding text columns")

dtr_a, dte, dtr_h = encode_features(dtr_a, dte, dtr_h)

print("Dropping away goals for away predictions")
X_a = dtr_a.drop(['FTAG'], axis=1)
y_a = dtr_a['FTAG']
print("Dropping home goals for home predictions")
X_h = dtr_h.drop(['FTHG'], axis=1)
y_h = dtr_h['FTHG']
print("Splitting for Home")
train_X_h, val_X_h, train_y_y, val_y_y = train_test_split(X_h, y_h, test_size= 0.2, random_state=1)
print("Splitting for away")
train_X, val_X, train_y, val_y = train_test_split(X_a, y_a, test_size= 0.2, random_state=1)
print("Running Random Forest model")
rf_model_on_full_data_a = RandomForestRegressor()
rf_model_on_full_data_h = RandomForestRegressor()
print("Fitting for away")
rf_model_on_full_data_a.fit(X_a, y_a)
print("Fitting for home")
rf_model_on_full_data_h.fit(X_h, y_h)
print("Predicting goals for Home Team")
test_preds_h = rf_model_on_full_data_h.predict(dte)
print("Predicting goals for Away Team")
test_preds_a = rf_model_on_full_data_a.predict(dte)
print("Output:")
output = pd.DataFrame({'Id': dte.index,
                       'FTHG': test_preds_h,
                       'FTAG': test_preds_a})
output.to_csv(r"C:\Users\harsh\Documents\My Dream\sportsintel.shop\Version 2\Output\Output.csv", index=False)
print(output)
print("Success!")
end = datetime.now()
time_taken = end - start
print('Time taken to complete: ', time_taken)
