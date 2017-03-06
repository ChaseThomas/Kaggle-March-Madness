import numpy as np
import pandas as pd
from logistic import LogReg
import tree

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
    
    team_data = reg_details_orig[reg_details_orig['Season'] > 2015]
    team_aggregate = teamProcessing(team_data)
    
    df_results = reg_details_orig[reg_details_orig['Season'] > 2010]
    
    logModel = LogReg()
    logModel.setup_model(df_results, team_aggregate)
    print logModel.get_prediction([1101, 1181], [1181, 1102])
    print logModel.get_prediction_int([1101, 1181], [1181, 1102])
    
    
    #treeModel = tree.derive_model(df_results)

main()