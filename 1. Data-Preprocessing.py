from sklearn.preprocessing import LabelEncoder
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
import pickle
import re
import json
import requests
import warnings
import pickle

warnings.filterwarnings('ignore')

seasons = [2014,2015,2016,2017,2018,2019,2020,2021,2022]
competitions = ['EPL','La_liga','Serie_A','Bundesliga','Ligue_1']

all_data = []
for season in seasons:
    for comp in competitions:
        url = f"https://understat.com/league/{comp}/{season}"
        html_doc = requests.get(url).text

        data = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", html_doc).group(1)
        data = re.sub(r'\\x([\dA-F]{2})', lambda g: chr(int(g.group(1), 16)), data)
        data = json.loads(data)

        for d in data:
            all_data.append({
                'season': season,
                'league': comp,
                'date': d['datetime'][:10], # first ten letters
                'home_team': d['h']['title'],
                'away_team': d['a']['title'],
                'home_goals': d["goals"]["h"],
                'away_goals': d["goals"]["a"],
                'home_xG':d['xG']['h'],
                'away_xG': d['xG']['a'],
                'forecast': list(d.get('forecast', {}).values())
            })

df_probs = pd.DataFrame(all_data)
# Split the forecast list into separate columns
df_probs[['home_win_prob', 'draw_prob', 'away_win_prob']] = df_probs['forecast'].apply(lambda x: pd.Series(x))

# Drop the original forecast column
df_probs = df_probs.drop('forecast', axis=1)

# Drop the games that haven't been played
df_probs = df_probs.dropna(how='any', subset=None)

# Mapping of old values to new values
rename_mapping = {
    'La_Liga': 'LaLiga',
    'Ligue_1': 'Ligue1',
    'Serie_A': 'SerieA'
}

# Replace values in the 'league' column using the mapping
df_probs['league'] = df_probs['league'].replace(rename_mapping)

# Store the DataFrame using pickle
with open('xg_win_probs.pickle', 'wb') as f:
    pickle.dump(df_probs, f)

df_probs.to_csv('understat_leagues_data.csv',index=False)



# ADVANCED XG STATS

data_directory = "C:/Users/paulc/Documents/Understat_xG/Data"


def combine_csv_by_row(directory):
    """
    Combines all excel files in a directory into a single DataFrame by row.

    Args:
    directory (str): Directory path containing CSV files.

    Returns:
    DataFrame: A single DataFrame with all CSVs appended by row.
    """
    global dataframe
    dataframe = pd.DataFrame()
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df['home_team'] = [i.split("--",1)[0] for i in df['fixture']]
            df['away_team'] = [i.split("--",1)[1] for i in df['fixture']]
            dataframe = pd.concat([dataframe, df], axis=0, ignore_index=True)
            dataframe['date'] = pd.to_datetime(dataframe['date'])
            dataframe = dataframe.sort_values(['date'],ascending=True).dropna(axis=1)
    return dataframe

df = combine_csv_by_row(data_directory)

if df.isna().sum().any():
    print("Some null rows encountered")
else:
    print("no nulls found")


df = df.drop(['fixture','home_np_xG','away_np_xG','home_sp_xG','away_sp_xG'],axis=1)

df = df[['date','season','league','home_team','away_team','home_directfk_xG', 'home_corner_xG', 'home_op_xG', 'home_pen_xG',
       'home_setpiece_xG', 'home_directfk_shots_ot', 'home_corner_shots_ot',
       'home_op_shots_ot', 'home_pen_shots_ot', 'home_setpiece_shots_ot',
       'home_directfk_shots', 'home_corner_shots', 'home_op_shots',
       'home_pen_shots', 'home_setpiece_shots', 'home_directfk_goals',
       'home_corner_goals', 'home_op_goals', 'home_pen_goals',
       'home_setpiece_goals', 'away_directfk_xG', 'away_corner_xG',
       'away_op_xG', 'away_pen_xG', 'away_setpiece_xG',
       'away_directfk_shots_ot', 'away_corner_shots_ot', 'away_op_shots_ot',
       'away_pen_shots_ot', 'away_setpiece_shots_ot', 'away_directfk_shots',
       'away_corner_shots', 'away_op_shots', 'away_pen_shots',
       'away_setpiece_shots','away_directfk_goals','away_corner_shots', 'away_op_shots', 'away_pen_shots',
       'away_setpiece_shots', 'date', 'away_directfk_goals',
       'away_corner_goals', 'away_op_goals', 'away_pen_goals',
       'away_setpiece_goals', 'home_xG', 'away_xG']]

# Trim whitespaces in 'home_team' and 'away_team' columns
df['home_team'] = df['home_team'].str.strip()
df['away_team'] = df['away_team'].str.strip()


columns_to_merge = ['season', 'league', 'home_team', 'away_team', 'home_win_prob', 'draw_prob', 'away_win_prob']
df = df.merge(df_probs[columns_to_merge], on=['season', 'league', 'home_team', 'away_team'])

# Store the DataFrame using pickle
with open('MaldiniNet.pickle', 'wb') as f:
    pickle.dump(df, f)

print('Cleaning completed and dataframe created!')