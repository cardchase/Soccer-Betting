import pandas as pd
from soccerapi.api import ApiBet365
from ast import literal_eval

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

api = ApiBet365()
print("Getting English Premier League Odds")
url_epl = 'https://www.bet365.it/#/AC/B1/C1/D13/E51761579/F2/'
odds_epl = api.odds(url_epl)
odds_epl = pd.DataFrame(data=odds_epl)
odds_epl['League'] = 'English Premier League'
print("Getting England Championship Odds")
url_ec1 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51791071/F2/'
odds_ec1 = api.odds(url_ec1)
odds_ec1 = pd.DataFrame(data=odds_ec1)
odds_ec1['League'] = 'England Championship'
print("Getting England League 1 Odds")
url_el1 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51791133/F2/'
odds_el1 = api.odds(url_el1)
odds_el1 = pd.DataFrame(data=odds_el1)
odds_el1['League'] = 'England League 1'
print("Getting England League 2 Odds")
url_el2 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51791197/F2/'
odds_el2 = api.odds(url_el2)
odds_el2 = pd.DataFrame(data=odds_el2)
odds_el2['League'] = 'England League 2'
print("Getting England National League Odds")
url_enl = 'https://www.bet365.it/#/AC/B1/C1/D13/E52803148/F2/'
odds_enl = api.odds(url_enl)
odds_enl = pd.DataFrame(data=odds_enl)
odds_enl['League'] = 'England National League'
print("Getting Belgium First Division Odds")
url_bfda = 'https://www.bet365.it/#/AC/B1/C1/D13/E51087843/F2/'
odds_bfda = api.odds(url_bfda)
odds_bfda = pd.DataFrame(data=odds_bfda)
odds_bfda['League'] = 'Belgium First Division'
print("Getting Germany Bundesliga 1 Odds")
url_gb1 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51669667/F2/'
odds_gb1 = api.odds(url_gb1)
odds_gb1 = pd.DataFrame(data=odds_gb1)
odds_gb1['League'] = 'Germany Bundesliga 1'
print("Getting Germany Bundesliga 2 Odds")
url_gb2 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51669781/F2/'
odds_gb2 = api.odds(url_gb2)
odds_gb2 = pd.DataFrame(data=odds_gb2)
odds_gb2['League'] = 'Germany Bundesliga 2'
print("Getting France League 1 Odds")
url_fl1 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51443718/F2/'
odds_fl1 = api.odds(url_fl1)
odds_fl1 = pd.DataFrame(data=odds_fl1)
odds_fl1['League'] = 'France League 1'
print("Getting France League 2 Odds")
url_fl2 = 'https://www.bet365.it/#/AC/B1/C1/D13/E51472414/F2/'
odds_fl2 = api.odds(url_fl2)
odds_fl2 = pd.DataFrame(data=odds_fl2)
odds_fl2['League'] = 'France League 2'
print("Getting Greece Super League 1 Odds")
url_gsl1 = 'https://www.bet365.it/#/AC/B1/C1/D13/E52334193/F2/'
odds_gsl1 = api.odds(url_gsl1)
odds_gsl1 = pd.DataFrame(data=odds_gsl1)
odds_gsl1['League'] = 'Greece Super League 1'
print("Getting Italy Serie A Odds")
url_seriea = 'https://www.bet365.it/#/AC/B1/C1/D13/E52224631/F2/'
odds_seriea = api.odds(url_seriea)
odds_seriea = pd.DataFrame(data=odds_seriea)
odds_seriea['League'] = 'Italy Serie A'
print("Getting Italy Serie B Odds")
url_serieb = 'https://www.bet365.it/#/AC/B1/C1/D13/E52547961/F2/'
odds_serieb = api.odds(url_serieb)
odds_serieb = pd.DataFrame(data=odds_serieb)
odds_serieb['League'] = 'Italy Serie B'
print("Getting Netherlands Eredivisie Odds")
url_nerd = 'https://www.bet365.it/#/AC/B1/C1/D13/E51695438/F2/'
odds_nerd = api.odds(url_nerd)
odds_nerd = pd.DataFrame(data=odds_nerd)
odds_nerd['League'] = 'Netherlands Eredivisie'
print("Getting Portugal Primeira Liga Odds")
url_ppl = 'https://www.bet365.it/#/AC/B1/C1/D13/E51878880/F2/'
odds_ppl = api.odds(url_ppl)
odds_ppl = pd.DataFrame(data=odds_ppl)
odds_ppl['League'] = 'Portugal Primeira Liga'
print("Getting Scotland Premiership Odds")
url_spr = 'https://www.bet365.it/#/AC/B1/C1/D13/E50681790/F2/'
odds_spr = api.odds(url_spr)
odds_spr = pd.DataFrame(data=odds_spr)
odds_spr['League'] = 'Scotland Premiership'
print("Getting Spain Primera Liga Odds")
url_spl = 'https://www.bet365.it/#/AC/B1/C1/D13/E52115687/F2/'
odds_spl = api.odds(url_spl)
odds_spl = pd.DataFrame(data=odds_spl)
odds_spl['League'] = 'Spain Primera Liga'
print("Getting Spain Segunda Odds")
url_sp2 = 'https://www.bet365.it/#/AC/B1/C1/D13/E52180402/F2/'
odds_sp2 = api.odds(url_sp2)
odds_sp2 = pd.DataFrame(data=odds_sp2)
odds_sp2['League'] = 'Spain Segunda'
print("Getting Turkey Super Lig Odds")
url_tsl = 'https://www.bet365.it/#/AC/B1/C1/D13/E52112990/F2/'
odds_tsl = api.odds(url_tsl)
odds_tsl = pd.DataFrame(data=odds_tsl)
odds_tsl['League'] = 'Turkey Super Lig'

odds_list = [odds_epl,
             odds_el1,
             odds_el2,
             odds_ec1,
             odds_enl,
             odds_bfda,
             odds_gb1,
             odds_gb2,
             odds_fl1,
             odds_fl2,
             odds_gsl1,
             odds_seriea,
             odds_serieb,
             odds_nerd,
             odds_ppl,
             odds_spr,
             odds_spl,
             odds_sp2,
             odds_tsl
             ]

df = pd.concat(odds_list)

df = df.dropna()

df.drop(list(df.filter(regex='None')), axis=1, inplace=True)

print("Rearranging columns")

df = df[['time', 'League', 'home_team', 'away_team', 'full_time_result', 'both_teams_to_score', 'double_chance']]

df.to_csv(r"C:\Users\Harshad\Documents\Project\Files\odds_raw.csv")

print("Bet365 Data downloaded successfully")
