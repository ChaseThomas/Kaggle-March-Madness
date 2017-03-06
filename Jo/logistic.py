import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#from sklearn.model_selection import GridSearchCV


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_dir = '../input/'
df_results_orig = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
df_results = df_results_orig[df_results_orig['Season'] > 2010]
df_resultsw = df_results[["Wfgm", "Wast", "Wdr", "Wstl", "Wto"]]
df_resultsw = df_resultsw.rename(index = str, columns = {"Wfgm":"fgm","Wast":"ast","Wdr":"dr","Wstl":"stl", "Wto":"to"})
df_resultsw["result"] = 1

df_resultsl = df_results[["Lfgm","Last", "Ldr", "Lstl", "Lto"]]
df_resultsl = df_resultsl.rename(index = str, columns = {"Lfgm":"fgm","Last":"ast","Ldr":"dr","Lstl":"stl", "Lto":"to"})
df_resultsl["result"] = 0

df_resultswl = pd.concat([df_resultsw, df_resultsl])
df_resultswl = df_resultswl.sample(frac=1)
df_resultswl.head()

#x = df_resultswl['fgm'].values.reshape(-1, 1)
#y = df_resultswl['result'].values.tolist()

logR = LogisticRegression()
logR.fit(df_resultswl.drop(['result'], axis=1), df_resultswl['result'])
'''
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(x.ravel(), y, color='black', zorder=20)

X_test = np.linspace(0, 300, 300)

print(logR.intercept_)

def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(X_test * logR.coef_[0][0] + logR.intercept_[0]).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)
'''
logR.predict_proba([[30,10, 20, 10, 2]])