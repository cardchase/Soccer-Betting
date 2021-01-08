print("Importing libraries")
from datetime import datetime

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tabulate import tabulate

start = datetime.now()
print("Importing libraries completed")
# Set option to display all the rows and columns in the dataset. If there are more rows, adjust number accordingly.
print("Set Option")
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print("Define Data Describe")

print("Getting files")
# Files
print("Reading Training Data")
data_train = pd.read_csv(r"C:\Users\Harshad\Documents\Project\Files\dataset_updated.csv", low_memory=False)
print("Reading Testing Data")
data_test = pd.read_excel(r"C:\Users\Harshad\Documents\Project\Files\Backtest.xlsx",
                          sheet_name='Testing', engine='openpyxl')
print("Reading files complete")

print("Keeping columns")
dtr = data_train
dte = data_test

dtr = dtr.loc[:, dtr.columns.intersection(
    ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR',
     'AR', 'B365H', 'B365D', 'B365A'])]

print("Dropping all NA")
dtr = dtr.dropna()
dte = dte.dropna()

data_test_pred = dte
# print("Splitting Date column")
dtr[['Date', 'Time']] = dtr['Date'].str.split(' ', expand=True)
dtr['Date'] = pd.to_datetime(dtr['Date'])
dtr['DOW'] = dtr['Date'].dt.dayofweek
dtr['Month'] = dtr['Date'].dt.month
dtr = dtr.loc[:, dtr.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
     'DOW', 'Month', 'B365H', 'B365D', 'B365A'])]

rcolumndict = {
    'HomeTeam': str,
    'AwayTeam': str,
    'FTHG': int,
    'FTAG': int,
    'B365H': float,
    'B365D': float,
    'B365A': float,
    'HS': int,
    'AS': int,
    'HST': int,
    'AST': int,
    'HF': int,
    'AF': int,
    'HC': int,
    'AC': int,
    'HY': int,
    'AY': int,
    'HR': int,
    'AR': int,
    'DOW': int,
    'Month': int
}
# print("Converting column types")
dtr = dtr.astype(rcolumndict)
# print("Using and splitting date columns to Day of week and month numbers for training data")
# dte['Date'] = dte['Date'].str.replace('/', '-')
dte['DateTime'] = pd.to_datetime(dte['DateTime'])
dte['DOW'] = dte['DateTime'].dt.dayofweek
dte['Month'] = dte['DateTime'].dt.month
dte_input = dte

# print("Adding NAN columns to Training Data")
dte = dte.reindex(
    columns=dte.columns.tolist() + ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR',
                                    'AR'])
# print("Keeping columns")
dte = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
     'DOW', 'Month', 'B365H', 'B365D', 'B365A'])]

dte_input = dte_input.loc[:, dte_input.columns.intersection(
    ['League', 'DateTime', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
     'HY', 'AY', 'HR', 'AR',
     'DOW', 'Month', 'B365H', 'B365D', 'B365A'])]

# print("Defining Team names from test dataset as string")
dte[['HomeTeam', 'AwayTeam']] = dte[['HomeTeam', 'AwayTeam']].astype(str)
# print("Converting Days to weekend or weekday match")
dtr['Weekend/Weekday'] = dtr['DOW'].apply(lambda x: 1 if (4 < x or x < 1) else 2)
dte['Weekend/Weekday'] = dte['DOW'].apply(lambda x: 1 if (4 < x or x < 1) else 2)
# print("Dropping DOW")


dtr = dtr.drop(['DOW'], axis=1)
dte = dte.drop(['DOW'], axis=1)


# print("Defining feature columns 'text = team names'")


def encode_features(df_train, df_test):
    features = ['HomeTeam', 'AwayTeam']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


# print("Encoding text columns")
dtr, dte = encode_features(dtr, dte)

print("Part: Predicting Goals")

dtr_g = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_g = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]

dtr_g_h = dtr_g.drop(['FTAG'], axis=1)
dtr_g_a = dtr_g.drop(['FTHG'], axis=1)
dte_g = dte_g.drop(['FTHG', 'FTAG'], axis=1)
dte_g = dte_g.dropna()

# print("Dropping home goals for home predictions")
X_g_h = dtr_g_h.drop(['FTHG'], axis=1)
y_g_h = dtr_g_h['FTHG']
# print("Dropping away goals for away predictions")
X_g_a = dtr_g_a.drop(['FTAG'], axis=1)
y_g_a = dtr_g_a['FTAG']
print("Splitting for Home")
train_X_g_h, val_X_g_h, train_y_g_h, val_y_g_h = train_test_split(X_g_h, y_g_h, test_size=0.2, random_state=1)
print("Splitting for away goals")
train_X_g_a, val_X_g_a, train_y_g_a, val_y_a = train_test_split(X_g_a, y_g_a, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_g_h = RandomForestRegressor()
rf_model_on_full_data_g_a = RandomForestRegressor()

print("Fitting for home goals")
rf_model_on_full_data_g_h.fit(X_g_h, y_g_h)
rf_model_on_full_data_g_a.fit(X_g_a, y_g_a)

print("Predicting goals for Home Team")
test_preds_h_g = rf_model_on_full_data_g_h.predict(dte_g)

print("Predicting goals for Away Team")
test_preds_a_g = rf_model_on_full_data_g_a.predict(dte_g)

print("Part: Predicting Corners")
dtr_c = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_c = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HC', 'AC', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_c_h = dtr_c.drop(['AC'], axis=1)
dtr_c_a = dtr_c.drop(['HC'], axis=1)
dte_c = dte_c.drop(['HC', 'AC'], axis=1)
dte_c = dte_c.dropna()
# print("Dropping home corners for home predictions")
X_c_h = dtr_c_h.drop(['HC'], axis=1)
y_c_h = dtr_c_h['HC']
# print("Dropping away corners for away predictions")
X_c_a = dtr_c_a.drop(['AC'], axis=1)
y_c_a = dtr_c_a['AC']
print("Splitting for Home")
train_X_c_h, val_X_c_h, train_y_c_h, val_y_c_h = train_test_split(X_c_h, y_c_h, test_size=0.2, random_state=1)
print("Splitting for Away")
train_X_c_a, val_X_c_a, train_y_c_a, val_c_a = train_test_split(X_c_a, y_c_a, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_c_h = RandomForestRegressor()
rf_model_on_full_data_c_a = RandomForestRegressor()
print("Fitting for home corners")
rf_model_on_full_data_c_h.fit(X_c_h, y_c_h)
rf_model_on_full_data_c_a.fit(X_c_a, y_c_a)
print("Predicting corners for Home Team")
test_preds_h_c = rf_model_on_full_data_c_h.predict(dte_c)
print("Predicting corners for Away Team")
test_preds_a_c = rf_model_on_full_data_c_a.predict(dte_c)

print("Part: Predicting Shots")
dtr_s = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_s = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_s_h = dtr_s.drop(['AS'], axis=1)
dtr_s_a = dtr_s.drop(['HS'], axis=1)
dte_s = dte_s.drop(['HS', 'AS'], axis=1)
dte_s = dte_s.dropna()
# print("Dropping home shots for home predictions")
X_s_h = dtr_s_h.drop(['HS'], axis=1)
y_s_h = dtr_s_h['HS']
# print("Dropping away shots for away predictions")
X_s_a = dtr_s_a.drop(['AS'], axis=1)
y_s_a = dtr_s_a['AS']
print("Splitting for Home")
train_X_s_h, val_X_s_h, train_y_s_h, val_y_s_h = train_test_split(X_s_h, y_s_h, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_s_h = RandomForestRegressor()
rf_model_on_full_data_s_a = RandomForestRegressor()
print("Fitting for shots")
rf_model_on_full_data_s_h.fit(X_s_h, y_s_h)
rf_model_on_full_data_s_a.fit(X_s_a, y_s_a)

print("Predicting shots for Home Team")
test_preds_h_s = rf_model_on_full_data_s_h.predict(dte_c)

print("Predicting shots for Away Team")
test_preds_a_s = rf_model_on_full_data_s_a.predict(dte_c)

print("Part: Predicting Shots on target")
dtr_st = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HST', 'AST', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_st = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HST', 'AST', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_st_h = dtr_st.drop(['AST'], axis=1)
dtr_st_a = dtr_st.drop(['HST'], axis=1)
dte_st = dte_st.drop(['HST', 'AST'], axis=1)
dte_st = dte_st.dropna()
X_st_h = dtr_st_h.drop(['HST'], axis=1)
y_st_h = dtr_st_h['HST']
X_st_a = dtr_st_a.drop(['AST'], axis=1)
y_st_a = dtr_st_a['AST']
print("Splitting for Home")
train_X_st_h, val_X_st_h, train_y_st_h, val_y_st_h = train_test_split(X_st_h, y_st_h, test_size=0.2, random_state=1)
print("Splitting for Away")
train_X_st_a, val_X_st_a, train_y_st_a, val_st_a = train_test_split(X_st_a, y_st_a, test_size=0.2, random_state=1)
rf_model_on_full_data_st_h = RandomForestRegressor()
rf_model_on_full_data_st_a = RandomForestRegressor()
print("Fitting for shots on target")
rf_model_on_full_data_st_h.fit(X_st_h, y_st_h)
rf_model_on_full_data_st_a.fit(X_st_a, y_st_a)
print("Predicting shots on target for Home Team")
test_preds_h_st = rf_model_on_full_data_st_h.predict(dte_st)
print("Predicting shots on target for Away Team")
test_preds_a_st = rf_model_on_full_data_st_a.predict(dte_st)

print("Part: Predicting Fouls")
dtr_f = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HF', 'AF', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_f = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HF', 'AF', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_f_h = dtr_f.drop(['AF'], axis=1)
dtr_f_a = dtr_f.drop(['HF'], axis=1)
dte_f = dte_f.drop(['HF', 'AF'], axis=1)
dte_f = dte_f.dropna()
# print("Dropping home fouls for home predictions")
X_f_h = dtr_f_h.drop(['HF'], axis=1)
y_f_h = dtr_f_h['HF']
# print("Dropping away fouls for away predictions")
X_f_a = dtr_f_a.drop(['AF'], axis=1)
y_f_a = dtr_f_a['AF']
print("Splitting for Home")
train_X_f_h, val_X_f_h, train_y_f_h, val_y_f_h = train_test_split(X_f_h, y_f_h, test_size=0.2, random_state=1)
print("Splitting for Away")
train_X_f_a, val_X_f_a, train_y_f_a, val_f_a = train_test_split(X_f_a, y_f_a, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_f_h = RandomForestRegressor()
rf_model_on_full_data_f_a = RandomForestRegressor()
print("Fitting for fouls")
rf_model_on_full_data_f_h.fit(X_f_h, y_f_h)
rf_model_on_full_data_f_a.fit(X_f_a, y_f_a)
print("Predicting fouls for Home Team")
test_preds_h_f = rf_model_on_full_data_f_h.predict(dte_f)
print("Predicting fouls for Away Team")
test_preds_a_f = rf_model_on_full_data_f_a.predict(dte_f)

print("Part: Predicting Yellow Cards")
dtr_y = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HY', 'AY', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_y = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HY', 'AY', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_y_h = dtr_y.drop(['AY'], axis=1)
dtr_y_a = dtr_y.drop(['HY'], axis=1)
dte_y = dte_y.drop(['HY', 'AY'], axis=1)
dte_y = dte_y.dropna()
# print("Dropping home yellow cards for home predictions")
X_y_h = dtr_y_h.drop(['HY'], axis=1)
y_y_h = dtr_y_h['HY']
# print("Dropping away yellow cards for away predictions")
X_y_a = dtr_y_a.drop(['AY'], axis=1)
y_y_a = dtr_y_a['AY']
print("Splitting for Home")
train_X_y_h, val_X_y_h, train_y_y_h, val_y_y_h = train_test_split(X_y_h, y_y_h, test_size=0.2, random_state=1)
print("Splitting for Away")
train_X_y_a, val_X_y_a, train_y_y_a, val_y_y_a = train_test_split(X_y_a, y_y_a, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_y_h = RandomForestRegressor()
rf_model_on_full_data_y_a = RandomForestRegressor()
print("Fitting for yellow cards")
rf_model_on_full_data_y_h.fit(X_y_h, y_y_h)
rf_model_on_full_data_y_a.fit(X_y_a, y_y_a)
print("Predicting yellow cards for Home Team")
test_preds_h_y = rf_model_on_full_data_y_h.predict(dte_y)
print("Predicting yellow cards for Away Team")
test_preds_a_y = rf_model_on_full_data_y_a.predict(dte_y)

print("Part: Predicting Red Cards")
dtr_r = dtr.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HR', 'AR', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dte_r = dte.loc[:, dte.columns.intersection(
    ['HomeTeam', 'AwayTeam', 'HR', 'AR', 'Weekend/Weekday', 'Month', 'B365H', 'B365D', 'B365A'])]
dtr_r_h = dtr_r.drop(['AR'], axis=1)
dtr_r_a = dtr_r.drop(['HR'], axis=1)
dte_r = dte_r.drop(['HR', 'AR'], axis=1)
dte_r = dte_r.dropna()
X_r_h = dtr_r_h.drop(['HR'], axis=1)
y_r_h = dtr_r_h['HR']
X_r_a = dtr_r_a.drop(['AR'], axis=1)
y_r_a = dtr_r_a['AR']
print("Splitting for Home")
train_X_r_h, val_X_r_h, train_y_r_h, val_y_r_h = train_test_split(X_r_h, y_r_h, test_size=0.2, random_state=1)
print("Splitting for Away")
train_X_r_a, val_X_r_a, train_y_r_a, val_y_r_a = train_test_split(X_r_a, y_r_a, test_size=0.2, random_state=1)
# print("Running Random Forest model")
rf_model_on_full_data_r_h = RandomForestRegressor()
rf_model_on_full_data_r_a = RandomForestRegressor()
print("Fitting for red cards")
rf_model_on_full_data_r_h.fit(X_r_h, y_r_h)
rf_model_on_full_data_r_a.fit(X_r_a, y_r_a)
print("Predicting red cards for Home Team")
test_preds_h_r = rf_model_on_full_data_r_h.predict(dte_r)
print("Predicting red cards for Away Team")
test_preds_a_r = rf_model_on_full_data_r_a.predict(dte_r)
print("Bringing it all together")

print(tabulate(dte_input, headers='keys'))

# print(tabulate(test_preds_h_g, headers='keys'))
#
# print(tabulate(test_preds_a_r, headers='keys'))

result = pd.DataFrame({
    'League': dte_input.League,
    'DateTime': dte_input.DateTime,
    'Home Team': dte_input.HomeTeam,
    'Away Team': dte_input.AwayTeam,
    'Full time Home Goals': test_preds_h_g,
    'Full time Away Goals': test_preds_a_g,
    'Home Team Corners': test_preds_h_c,
    'Away Team Corners': test_preds_a_c,
    'Home Team shots on goal': test_preds_h_s,
    'Away Team shots on goal': test_preds_a_s,
    'Home Team shots on target': test_preds_h_st,
    'Away Team shots on target': test_preds_a_st,
    'Home Team fouls': test_preds_h_f,
    'Away Team fouls': test_preds_a_f,
    'Home Team yellow cards': test_preds_h_y,
    'Away Team yellow cards': test_preds_a_y,
    'Home Team red cards': test_preds_h_r,
    'Away Team red cards': test_preds_a_r})

print(tabulate(result, headers='keys'))
# print(tabulate(dte_input, headers='keys'))
# print(tabulate(dte_g, headers='keys'))
result.to_csv(r"C:\Users\Harshad\Documents\Project\Files\output.csv")

output_dtypes = {'Full time Home Goals': int,
                 'Full time Away Goals': int,
                 'Home Team Corners': int,
                 'Away Team Corners': int,
                 'Home Team shots on goal': int,
                 'Away Team shots on goal': int,
                 'Home Team shots on target': int,
                 'Away Team shots on target': int,
                 'Home Team fouls': int,
                 'Away Team fouls': int,
                 'Home Team yellow cards': int,
                 'Away Team yellow cards': int,
                 'Home Team red cards': int,
                 'Away Team red cards': int}
result_con = result.astype(output_dtypes)
print(tabulate(result_con, headers='keys'))
result_con.to_csv(r"C:\Users\Harshad\Documents\Project\Files\output_int.csv")

print("Success!")
end = datetime.now()
time_taken = end - start
print('Time taken to complete: ', time_taken)
