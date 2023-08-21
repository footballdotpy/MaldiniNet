import pickle
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

def get_odd_or_text(td):
    if "data-odd" in td.attrs:
        return td["data-odd"]

    odd = td.select_one("[data-odd]")
    if odd:
        return odd["data-odd"]

    return td.get_text(strip=True) or 0

all_data = []

# Define the list of URLs to scrape
urls = ['https://www.betexplorer.com/football/england/premier-league-2022-2023/results/',
        'https://www.betexplorer.com/football/italy/serie-a-2022-2023/results/',
        'https://www.betexplorer.com/football/france/ligue-1-2022-2023/results/',
        'https://www.betexplorer.com/football/germany/bundesliga-2022-2023/results/',
        'https://www.betexplorer.com/football/spain/laliga-2022-2023/results/']


for url in urls:
    response = requests.get(url)
    if response.status_code == 404:
        continue
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for row in soup.select(".table-main tr:has(td)"):
        tds = [get_odd_or_text(td) if td else 0 for td in row.select("td")]
        round_ = row.find_previous("th").find_previous("tr").th.text
        all_data.append([round_, *tds])


df = pd.DataFrame(
    all_data, columns=["Round", "Match", "Score", "Avg_homeOdds", "Avg_drawOdds", "Avg_awayOdds", "Date"])

df['home_team'] = [i.split('-')[0] for i in df['Match']]
df['away_team'] = [i.split('-')[1] for i in df['Match']]

df['HomeGoals'] = [i.split(':', 1)[0] for i in df['Score']]
df['AwayGoals'] = [i.split(':', 1)[1] for i in df['Score']]

df = df[['home_team','away_team',"Avg_homeOdds", "Avg_drawOdds", "Avg_awayOdds","HomeGoals","AwayGoals"]]

df[["Avg_homeOdds", "Avg_drawOdds", "Avg_awayOdds","HomeGoals","AwayGoals"]] = df[["Avg_homeOdds", "Avg_drawOdds", "Avg_awayOdds","HomeGoals","AwayGoals"]].astype(float)

df['Overround'] = round((1/ df['Avg_homeOdds'] + 1 / df['Avg_drawOdds'] + 1 / df['Avg_awayOdds']) -1,5)

df['homeOdds_100%'] = round(1/(1/((3*df['Avg_homeOdds'])/(3-(df['Overround']*df['Avg_homeOdds'])))),2)
df['drawOdds_100%'] = round(1/(1/((3*df['Avg_drawOdds'])/(3-(df['Overround']*df['Avg_drawOdds'])))),2)
df['awayOdds_100%'] = round(1/(1/((3*df['Avg_awayOdds'])/(3-(df['Overround']*df['Avg_awayOdds'])))),2)

df = df[['home_team','away_team',"homeOdds_100%", "drawOdds_100%", "awayOdds_100%","HomeGoals","AwayGoals","Overround"]]

# Create a new column 'HomeWin' with 1 for home win, 0 otherwise
df['Home'] = np.where(df['HomeGoals'] > df['AwayGoals'], 1, 0)

# Create a new column 'Draw' with 1 for draw, 0 otherwise
df['Draw'] = np.where(df['HomeGoals'] == df['AwayGoals'], 1, 0)

# Create a new column 'AwayWin' with 1 for away win, 0 otherwise
df['Away'] = np.where(df['HomeGoals'] < df['AwayGoals'], 1, 0)

#remap team names to join on to other df

closing_odds_remap = {
                    'AC Ajaccio':'Ajaccio',
                    'Dortmund':'Borussia Dortmund',
                    'B. Monchengladbach':'Borussia M.Gladbach',
                    'Clermont':'Clermont Foot',
                    'FC Koln':'FC Cologne',
                    'Mainz':'Mainz 05',
                    'Manchester Utd':'Manchester United',
                    'Newcastle':'Newcastle United',
                    'Nottingham':'Nottingham Forest',
                    'Paris SG':'Paris Saint Germain',
                    'RB Leipzig':'RasenBallsport Leipzig',
                    'Schalke':'Schalke 04',
                    'Stuttgart':'VfB Stuttgart',
                    'Wolves':'Wolverhampton Wanderers'}

# iterate over columns
for key, value in df['home_team'].iteritems():
    df['home_team'] = df['home_team'].apply(lambda x: closing_odds_remap.get(x,x))

for key, value in df['away_team'].iteritems():
    df['away_team'] = df['away_team'].apply(lambda x: closing_odds_remap.get(x,x))

# Store the DataFrame using pickle
with open('closingOdds.pickle', 'wb') as f:
    pickle.dump(df, f)

df.to_csv('closingOdds.csv',index=False)

print('odds scraping completed and dataframe created!')