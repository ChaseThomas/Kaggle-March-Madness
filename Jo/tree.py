import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_dir = '../input/'
df_results_orig = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
df_results = df_results_orig[df_results_orig['Season'] > 2010]
df_results = df_results[["Wfgm", "Wast", "Wdr", "Wstl", "Wto", "Lfgm","Last", "Ldr", "Lstl", "Lto"]]
df_resultsw = df_results.rename(index = str, columns = {"Wfgm":"fgm1","Wast":"ast1","Wdr":"dr1","Wstl":"stl1", "Wto":"to1", "Lfgm":"fgm2","Last":"ast2","Ldr":"dr2","Lstl":"stl2", "Lto":"to2"})
df_resultsw["result"] = 1

df_resultsl = df_results.rename(index = str, columns = {"Wfgm":"fgm2","Wast":"ast2","Wdr":"dr2","Wstl":"stl2", "Wto":"to2", "Lfgm":"fgm1","Last":"ast1","Ldr":"dr1","Lstl":"stl1", "Lto":"to1"})

df_resultsl["result"] = 0

df_resultswl = pd.concat([df_resultsw, df_resultsl])
df_resultswl = df_resultswl.sample(frac=1)
df_resultswl.head()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_resultswl.drop(['result'], axis=1), df_resultswl['result'])

df_resultswl.drop(['result'], axis=1).sample(n = 1)

list = [10,12,10,10,30,20,10,10,10,10]

clf.predict_proba([list])