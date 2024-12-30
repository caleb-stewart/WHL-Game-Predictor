# WHL Game Outcome Prediction Model

This project utilizes machine learning to predict the outcome of WHL (Western Hockey League) hockey games. The model is based on historical team statistics and predicts which team is more likely to win an upcoming game. The goal is to use data-driven predictions to provide insights into game outcomes, though, as with all sports, predictions are inherently uncertain due to the unpredictable nature of human performance.

## Key Features:
- **Data Preparation**: Team statistics are updated with the most recent data, and various features like goals, power-play percentage (PP%), shots on goal (SOG), and faceoff win percentage (FOW%) are calculated.
- **Machine Learning Models**: The project uses four machine learning classification models: Random Forest, Naive Bayes, Logistic Regression, and Support Vector Classifier (SVC).
- **Model Evaluation**: The models are trained using historical team data, and each model is evaluated based on its accuracy. The model with the highest accuracy for each team is then used to predict the outcome of future games.
- **Prediction Process**: The model evaluates the home and away teams for an upcoming game and predicts the winner based on which classifier has the highest accuracy for each team. The accuracy for each model typically falls in the range of 65%-70%, which is a notable improvement over a 50-50 guess.

## Methodology:
### Feature Extraction:
The feature extraction process focuses on deriving relevant features from historical game data to train machine learning models. The key features used in this model are:

- **Goals Scored**: The total number of goals scored by the team and their opponent in previous games. This gives an indication of offensive performance.
- **Power Play Percentage (PP%)**: The percentage of successful power-play conversions. A key indicator of how effective a team is during power plays.
- **Shots on Goal (SOG)**: The number of shots on goal. More shots generally correlate with stronger offensive performance.
- **Faceoff Win Percentage (FOW%)**: The percentage of faceoffs won. This is an important metric for determining possession control during the game.

The model calculates the difference between the target team and opponent for each feature to capture the relative performance of the teams. This allows the classifier to consider both the absolute performance and how one team compares to the other in these areas.

For each team, the models are trained on historical data using the following steps:

1. **Data Preprocessing**: The features are scaled using `StandardScaler` to ensure that all features have equal weight in the model training process.
2. **Training and Testing**: The data is split into training and testing sets, typically with 80% used for training and 20% used for testing. Each classifier is trained on the training data and evaluated on the test data using accuracy scores.
3. **Model Evaluation**: For each classifier, accuracy scores are computed using confusion matrices. The classifier with the highest accuracy for each team is selected for future game predictions.

### Accuracy Evaluation:
The accuracy of each classifier is calculated using a confusion matrix to compare the predicted and actual outcomes of games. The classifier with the highest accuracy score for each team is chosen to make predictions for future matchups. Typically, the model achieves an accuracy of 65%-70%, which is a significant improvement over random guessing (50-50).

### Prediction:
Once the best classifier for each team has been selected, the model predicts the winner of future games by comparing the accuracy of the home and away teams' classifiers. The team with the higher accuracy is predicted to win. The model is trained and tested using real-time data (within the past game date) to ensure up-to-date predictions.

## API Requests:
The model relies on API requests to fetch the latest data on team performance and game statistics. These requests ensure that the model uses the most current data available, which is essential for making accurate predictions. The data includes statistics like goals scored, power-play percentage, shots on goal, and faceoff win percentage for each team.

## Necessary Files:
- **`whl_team_stats_script.ipynb`**: This notebook is used for the **initial setup** of team statistics. It fetches, processes, and stores the data in **`All_teams_WHL_stats.csv`**. Once the CSV is updated, this notebook does not need to be run again. An external user should not need to run this notebook
- **`All_teams_WHL_stats.csv`**: Contains historical data of WHL games, used for training the models. This file is automatically updated when running `whl_team_stats_script.ipynb`.
- **`whl_win_predictor.py`**: Main script that handles the model training, game prediction, and output generation.

## Using `whl_win_predictor.ipynb`:
The `whl_win_predictor.ipynb` notebook script is used to make predictions about the outcomes of upcoming games. It loads the historical data, trains the classifiers, and then uses the best performing classifier for each team to predict the winner of the next game. By evaluating the accuracy of the home and away team classifiers, the script identifies the more likely winner. Predictions are made using the latest available data.

The script can be run to generate predictions for the next game day. The predictions will include the predicted winner and the associated accuracy of the classifier used.

To run the script load it into your favorite IDE (I used Jupyter Labs) and simply run all code cells. The predictions will print at the bottom of the page

## Installation:
No need for library installations. The .ipynb files handle that for you with try/except statements.

## File Descriptions:
- **`All_teams_WHL_stats.csv`**: Contains historical data of WHL games, used for training the models.
- **`whl_win_predictor.py`**: Main script that handles the model training, game prediction, and output generation.
- **`predictions_output.txt`**: Stores the predicted game outcomes and model accuracy.

## Conclusion:
While the model isn't perfect, it provides valuable insights by predicting outcomes based on team statistics. It can assist in forecasting the next game day’s winner with an accuracy above random chance.
