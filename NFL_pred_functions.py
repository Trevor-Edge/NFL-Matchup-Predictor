from nfl_data_py import import_pbp_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#output: 'data'
def get_data():
    years = [2022, 2023, 2024]
    columns = [
        'game_id', 'home_team', 'away_team', 'posteam', 'defteam', 
        'passing_yards', 'rushing_yards', 'fumble_lost', 'interception', 
        'total_home_score', 'total_away_score', 'receiver_player_name'
    ]
    data = import_pbp_data(years, columns)

    data['receiver_player_name'] = data['receiver_player_name'].str.replace('Di.Johnson', 'Dio.Johnson')
    data = data.iloc[:, :11]
    return data

#output: 'model_data_output'
def model_data(data):
    ####### START OF DATA MANIPULATION FOR MODEL TRAINING DATA #######
    # Group offense stats
    offense_stats = data.groupby(['game_id', 'home_team', 'away_team', 'posteam']).agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'fumble_lost': 'sum',
        'interception': 'sum'
    }).reset_index()

    # Group defense stats
    defense_stats = data.groupby(['game_id', 'home_team', 'away_team', 'defteam']).agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'fumble_lost': 'sum',
        'interception': 'sum'
    }).reset_index()

    # Group scores
    scores = data.groupby(['game_id', 'home_team', 'away_team']).agg({
        'total_home_score': 'max',
        'total_away_score': 'max',
    }).reset_index()

    # Rename columns for clarity
    offense_stats = offense_stats.rename(columns={
        'game_id': 'Game', 'home_team': 'Home_Team', 'away_team': 'Away_Team', 
        'posteam': 'Team', 'passing_yards': 'Passing_YDS_Gained', 
        'rushing_yards': 'Rushing_YDS_Gained', 'fumble_lost': 'Fumbles', 
        'interception': 'Interceptions'
    })

    defense_stats = defense_stats.rename(columns={
        'game_id': 'Game', 'home_team': 'Home_Team', 'away_team': 'Away_Team', 
        'defteam': 'Team', 'passing_yards': 'Passing_YDS_Given', 
        'rushing_yards': 'Rushing_YDS_Given', 'fumble_lost': 'Fumbles_Taken', 
        'interception': 'Interceptions_Taken'
    })

    scores = scores.rename(columns={
        'game_id': 'Game', 'home_team': 'Home_Team', 'away_team': 'Away_Team', 
        'total_home_score': 'Home_Score', 'total_away_score': 'Away_Score'
    })

    # Combine datasets
    combined_stats = pd.merge(offense_stats, defense_stats, on=['Game', 'Home_Team', 'Away_Team', 'Team'])
    combined_stats2 = pd.merge(combined_stats, scores[['Game', 'Home_Score', 'Away_Score']], on='Game')

    combined_stats2 = combined_stats2[combined_stats2['Team'] == combined_stats2['Home_Team']].reset_index(drop=True)
    combined_stats2['winner'] = np.where(combined_stats2['Home_Score'] > combined_stats2['Away_Score'], 1, 0)

    # Add calculated columns
    combined_stats2['Giveaways'] = combined_stats2['Fumbles'] + combined_stats2['Interceptions']
    combined_stats2['Takeaways'] = combined_stats2['Fumbles_Taken'] + combined_stats2['Interceptions_Taken']

    # Drop unused columns
    combined_stats2 = combined_stats2.drop(columns=[
        'Fumbles', 'Interceptions', 'Fumbles_Taken', 'Interceptions_Taken', 
        'Team', 'Home_Score', 'Away_Score'
    ])
    combined_stats2['Giveaways_x'] = combined_stats2['Giveaways']
    combined_stats2['Takeaways_x'] = combined_stats2['Takeaways']

    # Rename columns
    combined_stats2 = combined_stats2.rename(columns={
        'Home_Team': 'Team1', 'Away_Team': 'Team2', 'Passing_YDS_Gained': 'Team1_Pass_YDS',
        'Rushing_YDS_Gained': 'Team1_Rush_YDS', 'Passing_YDS_Given': 'Team2_Pass_YDS',
        'Rushing_YDS_Given': 'Team2_Rush_YDS', 'Giveaways': 'Team1_Turnovers', 
        'Takeaways': 'Team1_Takeaways', 'Giveaways_x': 'Team2_Takeaways', 
        'Takeaways_x': 'Team2_Turnovers'
    })

    model_data_output = combined_stats2
    return model_data_output

#output: 'distribution_data_output'
def distribution_data(data):
    ####### START OF DATA MANIPULATION FOR DISTRIBUTION DATA #######
    nfl_teams = data['posteam'].unique().tolist()

    passing_statistics = data.groupby(['game_id', 'posteam'])[['passing_yards']].sum().reset_index()
    rushing_statistics = data.groupby(['game_id', 'posteam'])[['rushing_yards']].sum().reset_index()
    turnover_statistics = data.groupby(['game_id', 'posteam'])[['fumble_lost', 'interception']].sum().reset_index()
    takeaway_statistics = data.groupby(['game_id', 'defteam'])[['fumble_lost', 'interception']].sum().reset_index()
    
    results = []
    for team in nfl_teams:
        team_passing_data = passing_statistics[passing_statistics['posteam'] == team]
        team_rushing_data = rushing_statistics[rushing_statistics['posteam'] == team]
        team_turnover_data = turnover_statistics[turnover_statistics['posteam'] == team]
        team_takeaway_data = takeaway_statistics[takeaway_statistics['defteam'] == team]

        #Mean and Std of passing yds
        mean_passing_yds = team_passing_data['passing_yards'].mean()
        std_passing_yds = team_passing_data['passing_yards'].std()

        #Mean and Std of rushing yds
        mean_rushing_yds = team_rushing_data['rushing_yards'].mean()
        std_rushing_yds = team_rushing_data['rushing_yards'].std()

        #Mean and Std of team turnovers
        team_turnover_data['turnovers'] = team_turnover_data['fumble_lost'] + team_turnover_data['interception']
        mean_turnovers = team_turnover_data['turnovers'].mean()
        std_turnovers = team_turnover_data['turnovers'].std()
        
        #Mean and Std of team takeaways
        team_takeaway_data['takeaways'] = team_takeaway_data['fumble_lost'] + team_turnover_data['interception']
        mean_takeaways = team_takeaway_data['takeaways'].mean()
        std_takeaways = team_takeaway_data['takeaways'].std()

        results.append({
            'team': team,
            'mean_passing_yds': mean_passing_yds,
            'std_passing_yds': std_passing_yds,
            'mean_rushing_yds': mean_rushing_yds,
            'std_rushing_yds': std_rushing_yds,
            'mean_turnovers': mean_turnovers,
            'std_turnovers': std_turnovers,
            'mean_takeaways': mean_takeaways,
            'std_takeaways': std_takeaways
        })

    results = pd.DataFrame(results)
    distribution_data_output = results

    return distribution_data_output

#output: 'model', 'scaler'
def model_training(model_data):
    team1_features = ['Team1_Pass_YDS', 'Team1_Rush_YDS', 'Team1_Turnovers', 'Team1_Takeaways']
    team2_features = ['Team2_Pass_YDS', 'Team2_Rush_YDS', 'Team2_Turnovers', 'Team2_Takeaways']
    features = team1_features + team2_features

    # Target column
    target = 'winner'

    # Split into features (X) and labels (y)
    X = model_data[features].values
    y = model_data[target].values

    # Normalize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape for PyTorch
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, 1) #Singular output for binary classification

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    input_size = X_train.shape[1]
    model = LogisticRegressionModel(input_size)

    #Binary cross entropy loss for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) #Using Adam optimizer for better performance

    #Training Loop
    num_epochs = 300
    for epoch in range(num_epochs):
        #Forward Pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model = model
    scaler = scaler
    return model, scaler

#output: integer 1 or 0 (1 means team1 wins, 0 means team2 wins), team1/team2_stats come in list -> [passing, rushing, turnovers, takeaways]
def predict_winner(model, scaler, team1_stats, team2_stats):
    #Combine Stats
    combined_stats = team1_stats + team2_stats
    combined_stats = np.array(combined_stats).reshape(1,-1)

    #Scale features
    combined_stats_scaled = scaler.transform(combined_stats)

    #Convert to tensor
    input_tensor = torch.tensor(combined_stats_scaled, dtype=torch.float32)

    #Get model prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = (prediction >= 0.5).float().item()

    return int(predicted_class)


def game_output_generation(team1, team2, num_simulations=10000):

    #using get_data function to grab relevant data
    data = get_data()

    #using model_data to get training data
    model_data_output = model_data(data)

    #training model on model_data
    model, scaler = model_training(model_data_output)

    #using distribution_data to get distribution statistics to create normal distributions by team
    distribution_data_output = distribution_data(data)

    #pulling from distribution_data for each team
    team1_pass_mean = distribution_data_output.loc[distribution_data_output['team'] == team1, 'mean_passing_yds'].iloc[0]
    team1_pass_std = distribution_data_output.loc[distribution_data_output['team'] == team1, 'std_passing_yds'].iloc[0]
    team1_rush_mean = distribution_data_output.loc[distribution_data_output['team'] == team1, 'mean_rushing_yds'].iloc[0]
    team1_rush_std = distribution_data_output.loc[distribution_data_output['team'] == team1, 'std_rushing_yds'].iloc[0]
    team1_turnover_mean = distribution_data_output.loc[distribution_data_output['team'] == team1, 'mean_turnovers'].iloc[0]
    team1_turnover_std = distribution_data_output.loc[distribution_data_output['team'] == team1, 'std_turnovers'].iloc[0]
    team1_takeaway_mean = distribution_data_output.loc[distribution_data_output['team'] == team1, 'mean_takeaways'].iloc[0]
    team1_takeaway_std = distribution_data_output.loc[distribution_data_output['team'] == team1, 'std_takeaways'].iloc[0]

    team2_pass_mean = distribution_data_output.loc[distribution_data_output['team'] == team2, 'mean_passing_yds'].iloc[0]
    team2_pass_std = distribution_data_output.loc[distribution_data_output['team'] == team2, 'std_passing_yds'].iloc[0]
    team2_rush_mean = distribution_data_output.loc[distribution_data_output['team'] == team2, 'mean_rushing_yds'].iloc[0]
    team2_rush_std = distribution_data_output.loc[distribution_data_output['team'] == team2, 'std_rushing_yds'].iloc[0]
    team2_turnover_mean = distribution_data_output.loc[distribution_data_output['team'] == team2, 'mean_turnovers'].iloc[0]
    team2_turnover_std = distribution_data_output.loc[distribution_data_output['team'] == team2, 'std_turnovers'].iloc[0]
    team2_takeaway_mean = distribution_data_output.loc[distribution_data_output['team'] == team2, 'mean_takeaways'].iloc[0]
    team2_takeaway_std = distribution_data_output.loc[distribution_data_output['team'] == team2, 'std_takeaways'].iloc[0]

    #Caching the results
    simulation_results = []

    #Caching team1 values from random normal distributions
    team1_passing_predictions = []
    team1_rushing_predictions = []
    team1_turnovers_predictions = []
    team1_takeaways_predictions = []

    #Caching team2 values from random normal distributions
    team2_passing_predictions = []
    team2_rushing_predictions = []
    team2_turnovers_predictions = []
    team2_takeaways_predictions = []

    for i in range(num_simulations):
        team1_random_pass = np.random.normal(team1_pass_mean, team1_pass_std)
        team1_random_rush = np.random.normal(team1_rush_mean, team1_rush_std)
        team1_random_turnover = np.random.normal(team1_turnover_mean, team1_turnover_std)
        team1_random_takeaway = np.random.normal(team1_takeaway_mean, team1_takeaway_std)

        team2_random_pass = np.random.normal(team2_pass_mean, team2_pass_std)
        team2_random_rush = np.random.normal(team2_rush_mean, team2_rush_std)
        team2_random_turnover = np.random.normal(team2_turnover_mean, team2_turnover_std)
        team2_random_takeaway = np.random.normal(team2_takeaway_mean, team2_takeaway_std)

        team1_passing_predictions.append(team1_random_pass)
        team1_rushing_predictions.append(team1_random_rush)
        team1_turnovers_predictions.append(team1_random_turnover)
        team1_takeaways_predictions.append(team1_random_takeaway)
        
        team2_passing_predictions.append(team2_random_pass)
        team2_rushing_predictions.append(team2_random_rush)
        team2_turnovers_predictions.append(team2_random_turnover)
        team2_takeaways_predictions.append(team2_random_takeaway)

        team1_stats = [team1_random_pass, team1_random_rush, team1_random_turnover, team1_random_takeaway]
        team2_stats = [team2_random_pass, team2_random_rush, team2_random_turnover, team2_random_takeaway]

        winner = predict_winner(model, scaler, team1_stats, team2_stats)
        simulation_results.append(winner)

    team_results_dict = {
        'team1_passing_predictions': team1_passing_predictions,
        'team2_passing_predictions': team2_passing_predictions,
        'team1_rushing_predictions': team1_rushing_predictions,
        'team2_rushing_predictions': team2_rushing_predictions,
        'team1_turnovers_predictions': team1_turnovers_predictions,
        'team2_turnovers_predictions': team2_turnovers_predictions,
        'team1_takeaways_predictions': team1_takeaways_predictions,
        'team2_takeaways_predictions': team2_takeaways_predictions
    }

    return simulation_results, team_results_dict

team1 = 'KC'
team2 = 'CAR'
simulated_games, team_results_dict = game_output_generation(team1, team2)
KC_games = (sum(simulated_games)/10000)*100
CAR_games = ((10000-sum(simulated_games))/10000)*100
KC_games
CAR_games


### UPCOMING WEEK 12 GAMES ###

# Steelers vs Browns = predicted winner: Steelers win (67.8% to 32.2%) - Wrong
# Buccaneers vs Giants = predicted winner: Buccs win (57.9% to 42.1% - close matchup) - Right
# Chiefs vs Panthers = predicted winner: Chiefs win (69.5% to 30.5%) - Right
# Cowboys vs Commanders = predicted winner: Commanders win (83.3% to 16.2%) - Wrong
# Patriots vs Dolphins = predicted winner: Dolphins win (63.4% to 36.6%) - Right
# Titans vs Texans = predicted winner: Texans win (61.3% to 38.7%) - Wrong
# Lions vs Colts = predicted winner: Lions win (75.5% to 24.5%) - Right
# Vikings vs Bears = predicted winner: Vikings win (55.1% to 44.9% - close matchup) - Right
# Broncos vs Raiders = predicted winner: Broncos win (59.7% to 40.3% - close matchup) - Right
# 49ers vs Packers = predicted winner: 49ers win (60.2% to 39.8%) - Wrong
# Cardinals vs Seahawks = predicted winner: Cardinals win (64.0% to 36.0%) - Wrong
# Eagles vs Rams = predicted winner:  Eagles win (83.3% to 16.7%) - Right
# Ravens vs Chargers = predicted winner: Ravens win (65.5% to 34.5%) - Right

## 13 games - 8/13 Right, 5/13 Wrong ## 