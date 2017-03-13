import numpy as np
import pandas as pd
from logistic import LogReg
from tree import DeciTree
from itertools import product

data_dir = '~/Desktop/MM2017/'

def teamProcessing(data):
    df_resultsw = data[["Wteam", "Wfgm", "Wast", "Wdr", "Wstl", "Wto"]]
    df_resultsw = df_resultsw.rename(index = str, columns = {"Wteam": "team","Wfgm":"fgm","Wast":"ast","Wdr":"dr","Wstl":"stl", "Wto":"to"})

    df_resultsl = data[["Lteam","Lfgm","Last", "Ldr", "Lstl", "Lto"]]
    df_resultsl = df_resultsl.rename(index = str, columns = {"Lteam": "team","Lfgm":"fgm","Last":"ast","Ldr":"dr","Lstl":"stl", "Lto":"to"})
    
    data = pd.concat([df_resultsw, df_resultsl])

    grouped = data.groupby("team")
    grouped = grouped.aggregate(np.mean)
    
    return grouped

def pickRandomTrainingSet(df_results):
    return 0

def main():
    reg_details_orig = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
    tourney_hist = pd.read_csv(data_dir + 'TourneyDetailedResults.csv')
    tourney_2017 = pd.read_csv(data_dir + 'TourneySeeds.csv')
    
    team_data = reg_details_orig[reg_details_orig['Season'] > 2015]
    team_aggregate = teamProcessing(team_data)
    print(team_aggregate)
    
    df_results = reg_details_orig[reg_details_orig['Season'] > 2015]
    
    logModel = LogReg()
    logModel.setup_model(df_results, team_aggregate)
    
    tourney_hist['actual'] = 0
    tourney_hist.loc[tourney_hist['Wteam'] < tourney_hist['Lteam'], ['actual']] = 1
    should_swap = tourney_hist['Wteam'] > tourney_hist['Lteam']
    tourney_hist.loc[should_swap, ['Wteam', 'Lteam']] = tourney_hist.loc[should_swap, ['Lteam', 'Wteam']].values
    tourney_hist = tourney_hist.rename(columns={'Wteam':'team1',"Lteam":'team2'})
    
    #print(tourney_hist)
    
   # print(tourney_hist["team1"].tolist())
    
    #get_prediction(array of team1s, array of team2s)
    #print logModel.get_prediction(tourney_hist["team1"].tolist(), tourney_hist["team2"].tolist())
    #pred_results = logModel.get_prediction(tourney_hist["team1"].tolist(), tourney_hist["team2"].tolist())
    
    #tourney_hist['pred'] = pred_results
    
    #print(tourney_hist)
    
    #tourney_hist.to_csv(data_dir + "HistoricalPredictions.csv")
    
    #tourney_hist['pred'] = logModel.get_prediction_int(tourney_hist["team1"].tolist(), tourney_hist["team2"].tolist())
    
    #print(tourney_hist[tourney_hist.actual == tourney_hist.pred].sum())
    
    #treeModel = tree.derive_model(df_results)
    
    #tourney_results
    teamList = tourney_2017["Team"].tolist()
    tourney_pairings = pd.DataFrame(list(product(teamList, teamList)), columns=["team1", "team2"])
    keep = tourney_pairings['team1'] < tourney_pairings['team2']
    tourney_pairings = tourney_pairings.loc[keep,:]
    tourney_pairings["pred"] = logModel.get_prediction(tourney_pairings["team1"].tolist(), tourney_pairings["team2"].tolist())
    print(tourney_pairings)
    tourney_pairings.to_csv(data_dir + "FinalPredictions.csv")

main()