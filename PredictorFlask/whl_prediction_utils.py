import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_team_model(team_data):
    # Assuming team_data is a list of dictionaries with features and a target column

    # print(team_data[0][0].keys())
    df = pd.DataFrame(team_data, columns=['target_goals', 'opponent_goals', 'target_PP%', 'opponent_PP%', 'target_SOG', 'opponent_SOG', 'target_FOW%', 'opponent_FOW%', 'Home/Away', 'goals_diff', 'ppp_diff', 'sog_diff', 'fowp_diff', 'target_win'])

    # This is the dependency column we want to predict
    dependency = 'target_win'

    # Throw error if the dependency column is not present
    if dependency not in df.columns:
        raise ValueError(f"Missing {dependency} column.")

    # Get features and depencency columns
    X = df.drop(columns=[dependency])
    y = df[dependency]

    # Init scaler and standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers used
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(random_state=0),
        "SVC": SVC(kernel='rbf', probability=True, random_state=0)
    }

    # Train each classifier on scaled data
    for clf in classifiers.values():
        clf.fit(X_scaled, y)

    # Return the scaler used and trained classifiers
    return {
        "Scaler": scaler,
        "Classifiers": classifiers,
    }


def calculate_team_prob(team_data, models):
    '''
    team_data: LIST of one dictionary with features for the team to predict
    models: Dictionary containing the trained scaler and classifiers
    '''
    
    # Make sure the team_data is a DataFrame
    X = pd.DataFrame(team_data, columns=['target_goals', 'opponent_goals', 'target_PP%', 'opponent_PP%', 'target_SOG', 'opponent_SOG', 'target_FOW%', 'opponent_FOW%', 'Home/Away', 'goals_diff', 'ppp_diff', 'sog_diff', 'fowp_diff'])

    # Scale the prediction data using the scaler from the trained model
    # We only have one scaler for all classifiers
    X_scaled = models["Scaler"].transform(X)

    # Initialize total probability
    total_prob = 0

    # Iterate through each classifier and get the probability of winning
    for clf in models["Classifiers"].values():
        # [:, 1] is the probability of the positive class (win)
        prob = clf.predict_proba(X_scaled)[:, 1]
        # Add the probability to the total
        total_prob += prob[0]

    # Return the average probability of winning across all classifiers
    return total_prob / len(models["Classifiers"])