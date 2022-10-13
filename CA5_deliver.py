# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:34:08 2022

@author: Ivar
"""
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.plotting import scatterplotmatrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.plotting import scatterplotmatrix

df = pd.read_pickle('train.pkl')
df = df.replace('missing', np.nan) #Replaces missing with numpy nan and changes dtypes to float and int
df = pd.get_dummies(data=df, columns = ['Weather situation', 'Season'], drop_first = True)
df = df.fillna(df.median())

df_test = pd.read_pickle('test.pkl')

df_test = pd.get_dummies(data=df_test, columns = ['Weather situation', 'Season'], drop_first = True)
df_test = df_test.replace('missing', np.nan) #Replaces missing with numpy nan and changes dtypes to float and int

df_test = df_test.fillna(df_test.median())
df_test.head(11)

cols = df.columns[6:10]
scatterplotmatrix(df.iloc[:, 6: 10].values, figsize=(8, 8),
                     alpha=0.5, names = cols);
plt.tight_layout()

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

sns.violinplot(data=df.iloc[:, 6: 10], ax=axes[0])
sns.violinplot(data=df_test.iloc[:, 6: 10], ax= axes[1])
headlines = ['Train set', 'Test set']
for ax, header in zip(figure.axes, headlines):
    ax.tick_params(labelrotation=10)
    ax.set_title(header)
plt.show()

sns.violinplot(data=df['Rental bikes count'])
plt.title('Rental bikes count')
plt.show()

initial_size = df.shape

#for name in df.iloc[:, 6:10].columns:
#    df_strip[name] = df[name].between(0, 1)

#df.drop(df.loc[df["Rental bikes count"] < 700], axis=1)

#df = df[df['Rental bikes count'].between(0, 800)]
print(f'Initial shape:\t{initial_size}\nAfter removing:\t{df.shape}')

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

sns.violinplot(data=df.iloc[:, 6: 10], ax=axes[0])
sns.violinplot(data=df_test.iloc[:, 6: 10], ax= axes[1])
headlines = ['Train set', 'Test set']
for ax, header in zip(figure.axes, headlines):
    ax.tick_params(labelrotation=10)
    ax.set_title(header)
plt.show()

sns.violinplot(data=df['Rental bikes count'])
plt.title('Rental bikes count')
plt.show()

forest = RandomForestRegressor(n_estimators = 100,
                              random_state=1)

sfs1 = SFS(forest, 
           k_features=7, 
           forward=False, 
           floating=False, 
           verbose=0,
           scoring='r2',
           cv=3)

sfs1 = sfs1.fit(df. loc[:, df. columns!='Rental bikes count'], df['Rental bikes count'].values)
metricDict = sfs1.get_metric_dict()

fig1 = plot_sfs(metricDict, kind='std_dev')
#Try kind = {'std_dev', 'std_err', 'ci', None}

plt.title('Sequential Feature Selection (w. StdDev)')
plt.grid()
plt.show()

n_cols = 15
cols = list(metricDict[n_cols]['feature_names'])
print(*cols)

X = df. loc[:, df. columns!='Rental bikes count']
y = df['Rental bikes count'].values

X_cols = X[cols]
df_test_cols = df_test[cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


forest = RandomForestRegressor(n_estimators = 2000,
                              random_state=1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()
grad = GradientBoostingRegressor(
                         n_estimators      = 3000,   # 500 repetitions, called m in the presentation
                         learning_rate     = 0.01,   # Shrinkage of the classifier contribution (alpha).
                         random_state=1)

grad.fit(X, y)
pred = grad.predict(df_test)
pred[pred<0] = 3 #Changing predicted zero values to zero.

df_deliver = pd.DataFrame(pred, columns = ['Rental bikes count'])
df_deliver.to_csv('Deliverable_GradienBoostingClassifier01.csv')
forest = RandomForestRegressor(n_estimators = 2000,
                              random_state=1)
forest_pipe = make_pipeline(forest)

forest.fit(X, y)

pred = forest.predict(df_test)

df_deliver = pd.DataFrame(pred, columns = ['Rental bikes count'])
df_deliver.to_csv('Deliverable_randforestregressor.csv')


n_bins = np.arange(2, 32, 4)
bins_train_cut, labels = pd.cut(y_train,n_bins,
                                labels=False,
                               retbins=True)

#bins_train_qcut = pd.qcut(y_train,n_bins,  labels=False)
randforest = RandomForestClassifier(n_estimators = 2000)
pipe_forest = make_pipeline(randforest)
scores = []
for bins in n_bins:
    bins_train_cut, train_labels = pd.cut(y_train,bins,
                                labels=False,
                               retbins=True)
    pipe_forest.fit(X_train, bins_train_cut)
    pred = pipe_forest.predict(X_test)
    pred_reg = [train_labels[int(idx)] for idx in pred]
    scr = r2_score(y_test, pred_reg)
    scores.append(scr)

plt.scatter(n_bins, scores);

forest = RandomForestRegressor(n_estimators = 2000,
                              random_state=22)


forest.fit(X, y)

pred = forest.predict(df_test)
pred[pred<0] = 3 #Make all neg values equal 3.

df_deliver = pd.DataFrame(pred, columns = ['Rental bikes count'])
df_deliver.to_csv('Deliverable_randforestregressor.csv')
