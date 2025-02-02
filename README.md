# NFL Matchup Predictor

According to this model, the Eagles will be winning Super Bowl LIX, but how can you bet against the Chiefs???:) This project spins up an interactive web application that predicts NFL game outcomes using machine learning. The system analyzes historical play-by-play data and runs 10,000 simulations to generate win predictions for any two NFL teams.

## Features

- Interactive UI with NFL team images organized by division
- Real-time game outcome predictions
- Machine learning model trained on NFL play-by-play data
- 10,000 simulations per prediction
- Statistical analysis including:
  - Passing yards
  - Rushing yards
  - Turnovers
  - Takeaways

## Tech Stack

- **Backend:**
  - Python 3.x
  - Flask
  - PyTorch (Machine Learning)
  - pandas
  - numpy
  - scikit-learn
  - nfl_data_py (NFL data API)

- **Frontend:**
  - HTML
  - CSS
  - JavaScript

## Installation

1. Clone the repository:

git clone [https://github.com/Trevor-Edge/NFL-Matchup-Predictor.git]

cd NFL-Matchup-Predictor

2. Install required Python packages:

pip install -r requirements.txt

3. Ensure all team logos are in the correct directories under `static/`:

static/

├── AFC_North/

├── AFC_South/

├── AFC_East/

├── AFC_West/

├── NFC_North/

├── NFC_South/

├── NFC_East/

├── NFC_West/

└── style.css

## Project Structure

NFL-Matchup-Predictor/

├── app.py                 # Flask application main file

├── NFL_pred_functions.py  # ML model and prediction functions

├── static/               # Static assets

│   ├── style.css        # Main stylesheet

│   └── */               # Team logo directories

├── templates/           # HTML templates

│   └── index.html      # Main page template

└── README.md

## Usage

1. Start the Flask server:

flask run

2. Open your web browser and navigate to `http://localhost:5000`

3. Select two NFL teams by clicking on their logos

4. View the prediction results, which will show:
   - Win probability for each team
   - Based on 10,000 simulations

## How It Works

1. **Data Collection:**
   - Fetches NFL play-by-play data using nfl_data_py
   - Processes historical game statistics

2. **Model Training:**
   - Uses PyTorch for machine learning
   - Trains on historical game data
   - Features include passing yards, rushing yards, turnovers, and takeaways

3. **Prediction Process:**
   - Runs 10,000 simulations sampling from historical game data distributions for each statistic (passing yards, rushing yards, turnovers, takeaways)
   - Generates random normal distributions based on team statistics
   - Predicts winner for each simulation
   - Calculates final win probabilities

## Model Performance

The model's predictions are based on 2022-2024 season data and have shown promising results. In a recent test (Week 12), the model correctly predicted 8 out of 13 game outcomes correctly with just 4 simple statistics.

Week 12 was a tough one as the 49ers were without starting QB Brock Purdy and the Browns/Steelers game was played in the SNOW! These were two of five incorrect predictions but looking towards the future, updates will be made making more model improvements such as including more relevant features, testing different models, and more!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

NFL data provided by nfl_data_py
