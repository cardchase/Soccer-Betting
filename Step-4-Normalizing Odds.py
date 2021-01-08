import pandas as pd
from ast import literal_eval

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tabulate import tabulate

print("Reading raw file")
df = pd.read_csv(r"C:\Users\Harshad\Documents\Project\Files\odds_raw.csv")

print("Rearranging columns")
df = df[['time', 'League', 'home_team', 'away_team', 'full_time_result', 'both_teams_to_score', 'double_chance']]

df.iloc[:, 4:] = df.iloc[:, 4:].applymap(literal_eval)

df_list = list()

for col in df.columns[4:]:
    v = pd.json_normalize(df[col])
    v.columns = [f'{col}_{c}' for c in v.columns]
    df_list.append(v)

df_normalized = pd.concat([df.iloc[:, :4]] + df_list, axis=1)
df_normalized.time = pd.to_datetime(df_normalized.time)

df_normalized=df_normalized.dropna()

integers = {
    'full_time_result_1': int,
    'full_time_result_X': int,
    'full_time_result_2': int,
    'both_teams_to_score_yes': int,
    'both_teams_to_score_no': int,
    'double_chance_1X': int,
    'double_chance_12': int,
    'double_chance_2X': int
}
df_normalized = df_normalized.astype(integers)

df_normalized = df_normalized.loc[:, df_normalized.columns.intersection(
    ['time', 'League', 'home_team', 'away_team', 'full_time_result_1', 'full_time_result_X', 'full_time_result_2'])]

df_normalized = df_normalized.set_axis(['DateTime', 'League', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A'],
                                       axis=1, inplace=False)

df_normalized['B365H'] = df_normalized['B365H'].div(1000)
df_normalized['B365D'] = df_normalized['B365D'].div(1000)
df_normalized['B365A'] = df_normalized['B365A'].div(1000)
print("Writing to excel")


def write_excel(filename, sheetname, dataframe):
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
        workBook = writer.book
        try:
            workBook.remove(workBook[sheetname])
        except:
            print("Worksheet does not exist")
        finally:
            dataframe.to_excel(writer, sheet_name=sheetname, index=False)
            writer.save()


write_excel(r"C:\Users\Harshad\Documents\Project\Files\Backtest.xlsx", 'BET365', df_normalized)
print(df_normalized)

print("Reconciling with team dictionary names")

df_odds = df_normalized

df_dict = pd.read_excel(r"C:\Users\Harshad\Documents\Project\Files\Backtest.xlsx", sheet_name='Team Dict',  engine='openpyxl')

df_dict = df_dict.dropna()

s = df_dict.set_index('Bet365_Names')['Team (Dataset)']
df_odds['HomeTeam'] = df_odds['HomeTeam'].map(s)
df_odds['AwayTeam'] = df_odds['AwayTeam'].map(s)

print(tabulate(df_odds, headers='keys'))

write_excel(r"C:\Users\Harshad\Documents\Project\Files\Backtest.xlsx", 'Testing', df_normalized)
