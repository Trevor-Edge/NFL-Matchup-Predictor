from flask import Flask, render_template, jsonify, request
from nfl_data_py import import_pbp_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import importlib
import NFL_pred_functions
importlib.reload(NFL_pred_functions)

from NFL_pred_functions import game_output_generation

app = Flask(__name__, static_folder='static')

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to run your_script.py and return its output
@app.route('/fetch-data', methods=['POST'])
def fetch_data():
    try:
        data = request.get_json() # Get JSON data from the request
        teams = data.get('teams')

        if not teams or len(teams) != 2:
            return jsonify({'error': 'You must select exactly two teams.'}), 400
        
        # Process the teams (e.g., fetch data or perform operations)
        simulation_results, team_results_dict = game_output_generation(teams[0], teams[1])
        team1_wins = (sum(simulation_results)/10000)*100
        team2_wins = ((10000 - sum(simulation_results))/10000)*100
        
        # Process the teams (e.g., fetch data or perform operations)
        # Example response:
        response = {
            'message': f'{teams[0]} wins {round(team1_wins, 1)}% of simulations while {teams[1]} wins {round(team2_wins,1)}% of simulations.',
            'team-data': {'team1': teams[0], 'team2': teams[1]},
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
if __name__ == '__main__':
    app.run(debug=True)