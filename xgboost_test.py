# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:13:50 2019

@author: elisalvo
"""
import pandas as pd
from pandas import *
import os
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from IPython.display import HTML

os.getcwd()
"""
path='C:/Pasta_Python/'
os.chdir(path)
"""
dados = pd.read_csv('E:\\base_python.csv')
dados.head()
dados.info()
dados.count()
len(dados)
print(dados['pesocat'].describe())

plt.figure(figsize=(9,8))
sns.distplot(dados['peso'], color='g', bins = 1, hist_kws={'alpha': 0.4});
dados_num = dados.select_dtypes(include = ['float64', 'int64'])
dados_num.head()
dados_num.hist(figsize=(16,20), bins = 50, xlabelsize = 8, ylabelsize =8);

dados_x = dados.drop(columns = ['pesocat'])
dados_x.head()
dados_y = dados['pesocat']
dados_y.head()
HTML(dados_x.to_html())

modelo = final_GBM.fit(dados_x.values, y_train.values)

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
dtrain = xgb.DMatrix (x_train, label = y_train)

xd = dados[: ,1:184]
yd = dados[:,0]

model = XGBClassifier()
model.fit(xd,yd)

plot_importance(model)
pyplot.show()

