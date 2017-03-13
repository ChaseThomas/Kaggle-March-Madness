from sklearn.model_selection import train_test_split
from ryan.preprocess import load_tourney_results
from ryan.preprocess import preprocess_team_avg
from ryan.preprocess import dataframes_to_matricies
from ryan.preprocess import load_toruney_seeds
from ryan.preprocess import preprocess_massey
from ryan.preprocess import preprocess_massey_2017
from ryan.logistic_classifier import LogisticClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np


SEED = 31337


def main():
    massey_ordinals_df = preprocess_massey()
    tourney_results_df = load_tourney_results()
    tourney_seeds_df = load_toruney_seeds()
    team_avgs_df = preprocess_team_avg()

    classifier = LogisticClassifier(
        np.empty([0, 14]), np.empty([0,]),
        load_model="ryan/saved-networks/LogisticClassifier-2017-03-11_21-03-54.ckpt",
        num_epochs=0, beta=0.05, seed=SEED
    )

    predictions_df = pd.DataFrame(columns=['team1', 'team2', 'season', 'predicted', 'actual'])
    future_predictions_df = pd.DataFrame(columns=['team1', 'team2', 'predicted'])
    for season in (2014, 2015, 2016, 2017):
        tourney_df = tourney_seeds_df
        tourney_df = tourney_df[tourney_df['Season'] == season]
        del tourney_df['Season']
        tourney_df = tourney_df.sort_values('Team').reset_index(drop=True)
        for idx1, row1 in tourney_df.iterrows():
            team1 = row1['Team']
            seed1 = row1['Seed']
            for idx2, row2 in tourney_df[tourney_df.index > idx1].iterrows():
                team2 = row2['Team']
                seed2 = row2['Seed']
                features = get_features(massey_ordinals_df, team_avgs_df, team1, team2, seed1, seed2, season)
                prediction = classifier.predict_values(features)[0][0]
                if season == 2017:
                    future_predictions_df = future_predictions_df.append([{
                        'team1': team1, 'team2': team2, 'predicted': prediction
                    }], ignore_index=True)
                    continue
                tmp = tourney_results_df.loc[(tourney_results_df['Season'] == season)]
                tmp1 = tmp.loc[(tmp['Wteam'] == team1) & (tmp['Lteam'] == team2)]
                tmp2 = tmp.loc[(tmp['Wteam'] == team2) & (tmp['Lteam'] == team1)]
                if not tmp1.empty:
                    actual = 1
                elif not tmp2.empty:
                    actual = 0
                else:
                    continue
                predictions_df = predictions_df.append([{
                    'team1': team1, 'team2': team2, 'season': season, 'predicted': prediction, 'actual': actual
                }], ignore_index=True)

    predictions_df.set_index(['team1', 'team2', 'season']).to_csv("ryan/historical_predictions.csv")
    future_predictions_df.set_index(['team1', 'team2']).to_csv("ryan/future_predictions.csv")
    predictions_2017 = predictions_df.loc[predictions_df['season'] == 2017]

'''
    x_matrix, y_matrix = dataframes_to_matricies(team_avgs_df, massey_ordinals_df, tourney_seeds_df, tourney_results_df)
    # Split data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.3, random_state=SEED
    )

    print("Training size: %d, Testing size: %d" % (x_train.shape[0], x_test.shape[0]))

    print("Beginning Logistic Classifier Demo")
    classifier = LogisticClassifier(
        x_train, y_train,
        load_model="ryan/saved-networks/LogisticClassifier-2017-03-11_21-03-54.ckpt",
        num_epochs=0, beta=0.05, seed=SEED
    )
    print("Test Accuracy: %f" % classifier.test_accuracy(x_test, y_test))

    #classifier.save_model()
'''


def get_features(massey_ordinals, team_avgs, team1, team2, seed1, seed2, season):
    team1_massey = massey_ordinals.loc[(massey_ordinals['season'] == season) & (massey_ordinals['team'] == team1)]['orank'].iloc[0]
    team2_massey = massey_ordinals.loc[(massey_ordinals['season'] == season) & (massey_ordinals['team'] == team2)]['orank'].iloc[0]
    team1_avgs = team_avgs.loc[(team_avgs['Season'] == season) & (team_avgs['Team'] == team1)]
    del team1_avgs['Season']
    del team1_avgs['Team']
    team2_avgs = team_avgs.loc[(team_avgs['Season'] == season) & (team_avgs['Team'] == team2)]
    del team2_avgs['Season']
    del team2_avgs['Team']
    team1_avgs = team1_avgs.values
    team2_avgs = team2_avgs.values
    return np.concatenate(([[team1_massey - team2_massey]], [[seed1 - seed2]], team1_avgs - team2_avgs), axis=1)


if __name__ == "__main__":
    main()