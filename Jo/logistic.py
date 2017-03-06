import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def derive_model(df_results):
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
    
    return logR
