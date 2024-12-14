import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


df=pd.read_csv("train.csv", sep=",")


df.head()


df.info()


print(df.columns)


estacion = df['substation'].unique()

print(estacion)


df.info()


dfs_AJAHUEL = df[df['substation'] == 'AJAHUEL']
dfs_BUIN = df[df['substation'] == 'BUIN']
dfs_CHENA = df[df['substation'] == 'CHENA']
dfs_CNAVIA = df[df['substation'] == 'CNAVIA']
dfs_AJAHUEL = df[df['substation'] == 'AJAHUEL']
dfs_BUIN = df[df['substation'] == 'BUIN']
dfs_CHENA = df[df['substation'] == 'CHENA']
dfs_CNAVIA = df[df['substation'] == 'CNAVIA']
dfs_ELSALTO = df[df['substation'] == 'ELSALTO']
dfs_FLORIDA = df[df['substation'] == 'FLORIDA']
dfs_LOSALME = df[df['substation'] == 'LOSALME']


y_ajahuel = dfs_AJAHUEL['consumption'].to_numpy()

y_ajahuel

x_ajauhuel = dfs_AJAHUEL['date'].to_numpy()

x_ajauhuel

x_ajauhuel.shape


x_buin = dfs_BUIN['date'].to_numpy()


x_buin

x_buin.shape


plt.plot(x_ajauhuel,y_ajahuel)