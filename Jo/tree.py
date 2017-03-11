import numpy as np
import pandas as pd
from sklearn import tree


class DeciTree():
    def log5(self, pa, pb):
        return (pa-pa*pb)/(pa+pb-2*pa*pb)
    
    def get_prediction(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        team1_data = team1_data.reset_index();
        team2_data = team2_data.reset_index();
        
        team1_data = team1_data.rename(columns = {"fgm":"fgm1","ast":"ast1","dr":"dr1","stl":"stl1", "to":"to1"}).drop('team', 1)
        team2_data = team2_data.rename(columns = {"fgm":"fgm2","ast":"ast2","dr":"dr2","stl":"stl2", "to":"to2"}).drop('team', 1)
        
        combined = pd.concat([team1_data, team2_data], axis = 1)
        
        return self.model.predict_proba(combined)[:,1]
    
    def get_prediction_int(self, team1, team2):
        team1_data = self.team_data.loc[team1, :]
        team2_data = self.team_data.loc[team2, :]
        
        team1_data = team1_data.reset_index();
        team2_data = team2_data.reset_index();
        
        team1_data = team1_data.rename(columns = {"fgm":"fgm1","ast":"ast1","dr":"dr1","stl":"stl1", "to":"to1"}).drop('team', 1)
        team2_data = team2_data.rename(columns = {"fgm":"fgm2","ast":"ast2","dr":"dr2","stl":"stl2", "to":"to2"}).drop('team', 1)
        
        combined = pd.concat([team1_data, team2_data], axis = 1)
        return self.model.predict(combined)
     
    def setup_model(self, df_results, team_data):
        df_results = df_results[["Wfgm", "Wast", "Wdr", "Wstl", "Wto", "Lfgm","Last", "Ldr", "Lstl", "Lto"]]
        df_resultsw = df_results.rename(index = str, columns = {"Wfgm":"fgm1","Wast":"ast1","Wdr":"dr1","Wstl":"stl1", "Wto":"to1", "Lfgm":"fgm2","Last":"ast2","Ldr":"dr2","Lstl":"stl2", "Lto":"to2"})
        df_resultsw["result"] = 1
        
        df_resultsl = df_results.rename(index = str, columns = {"Wfgm":"fgm2","Wast":"ast2","Wdr":"dr2","Wstl":"stl2", "Wto":"to2", "Lfgm":"fgm1","Last":"ast1","Ldr":"dr1","Lstl":"stl1", "Lto":"to1"})
        df_resultsl["result"] = 0
        
        df_resultswl = pd.concat([df_resultsw, df_resultsl])
        df_resultswl = df_resultswl.sample(frac=1)
        
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(df_resultswl.drop(['result'], axis=1), df_resultswl['result'])
        
        self.model = clf
        self.team_data = team_data