import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dask.dataframe.core import DataFrame
from bokeh.charts.utils import df_from_json

class LogReg():
    #def __init__(self):
        #self.
        
    def log5(self, pa, pb):
        return (pa-pa*pb)/(pa+pb-2*pa*pb)
    
    def get_prediction(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        team1_data = team1_data.reset_index()
        team2_data = team2_data.reset_index()
        
        bob = team1_data - team2_data
        bob = bob.drop(axis = 1, labels = "team")
        
        pbob = self.model.predict_proba(bob)[:, 1]
        
        return pbob
    
    def get_prediction_int(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        team1_data = team1_data.reset_index()
        team2_data = team2_data.reset_index()
        
        bob = team1_data - team2_data
        bob = bob.drop(axis = 1, labels = "team")
        
        pbob = self.model.predict_proba(bob)[:, 1]
        
        pbob = pbob.round(0)
        
        return pbob
     
    def setup_model(self, df_results, team_data):
        
        df_resultsw = df_results[["Wfgm", "Wast", "Wdr", "Wstl", "Wto"]]
        #df_resultsw = df_resultsw.reindex()
        df_resultsw = df_resultsw.rename(index = str, columns = {"Wfgm":"fgm","Wast":"ast","Wdr":"dr","Wstl":"stl", "Wto":"to"})
        df_resultsl = df_results[["Lfgm","Last", "Ldr", "Lstl", "Lto"]]
        #df_resultsl = df_resultsl.reindex()
        df_resultsl = df_resultsl.rename(index = str, columns = {"Lfgm":"fgm","Last":"ast","Ldr":"dr","Lstl":"stl", "Lto":"to"})
        
        df_wins = df_resultsw - df_resultsl
        df_wins["result"] = 1
        df_losses = df_resultsl - df_resultsw
        df_losses["result"] = 0
        
        df_diffs = pd.concat([df_wins, df_losses])
        df_diffs = df_diffs.sample(frac=1)
        
        #df_resultsw["result"] = 1
        
        #df_resultsl["result"] = 0
        
        #df_resultswl = pd.concat([df_resultsw, df_resultsl])
        #df_resultswl = df_resultswl.sample(frac=1)
        
        print(df_diffs)
        
        logR = LogisticRegression()
        logR = logR.fit(df_diffs.drop(['result'], axis=1), df_diffs['result'])
        
        self.model = logR
        self.team_data = team_data
