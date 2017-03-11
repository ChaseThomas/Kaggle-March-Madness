import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class LogReg():
    #def __init__(self):
        #self.
        
    def log5(self, pa, pb):
        return (pa-pa*pb)/(pa+pb-2*pa*pb)
    
    def get_prediction(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        print team1_data
        
        pa = self.model.predict_proba(team1_data)[:, 1]
        pb = self.model.predict_proba(team2_data)[:, 1]
        
        return self.log5(pa, pb)
    
    def get_prediction_int(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        pa = self.model.predict_proba(team1_data)[:, 1]
        pb = self.model.predict_proba(team2_data)[:, 1]
        
        probs = self.log5(pa, pb)
        
        probs = probs.round(0)
        
        return probs
     
    def setup_model(self, df_results, team_data):
        df_resultsw = df_results[["Wfgm", "Wast", "Wdr", "Wstl", "Wto"]]
        
        df_resultsw = df_resultsw.rename(index = str, columns = {"Wfgm":"fgm","Wast":"ast","Wdr":"dr","Wstl":"stl", "Wto":"to"})
        df_resultsw["result"] = 1
        
        df_resultsl = df_results[["Lfgm","Last", "Ldr", "Lstl", "Lto"]]
        df_resultsl = df_resultsl.rename(index = str, columns = {"Lfgm":"fgm","Last":"ast","Ldr":"dr","Lstl":"stl", "Lto":"to"})
        df_resultsl["result"] = 0
        
        df_resultswl = pd.concat([df_resultsw, df_resultsl])
        df_resultswl = df_resultswl.sample(frac=1)
        
        logR = LogisticRegression()
        logR = logR.fit(df_resultswl.drop(['result'], axis=1), df_resultswl['result'])
        
        self.model = logR
        self.team_data = team_data
