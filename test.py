# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:41:01 2024

@author: ivoto
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# se carga el dataset, con separados ;, se transforman las dos columnas Date y Time a una única columna con tipo datetime
# en el fichero CSV existen NaN identificados por ?
df = pd.read_csv('household_power_consumption.txt', sep=';',
                        parse_dates={'dt' : ['Date', 'Time']}, 
                        infer_datetime_format=True,low_memory=False,
                        na_values=['nan','?'])
df.dropna(inplace=True)
dataFrame = df.drop(columns='dt')
dataFrame = dataFrame.head(500000)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(dataFrame) 

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 21):
     kmeans = KMeans(n_clusters = i, init = "k-means++", algorithm='lloyd', n_init=10 ,random_state = 0)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1,21), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


n = 4
kmeans = KMeans(n_clusters = n).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

kmeansmodel = KMeans(n_clusters = n, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)



import plotly.express as px

clusters = pd.DataFrame(X,columns=X.columns)
clusters['label'] = kmeansmodel.labels_
polar = clusters.groupby("label").mean().reset_index()
polar = pd.melt(polar,id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=600)
fig4.show()

import warnings

# Ignorar FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Tu código aquí
import plotly.express as px
import pandas as pd

columns_names = ['col1', 'col2', 'col3', 'col4','col5', 'col6','col7']
clusters = pd.DataFrame(X, columns=columns_names)
clusters['label'] = kmeansmodel.labels_
polar = clusters.groupby("label").mean().reset_index()
polar = pd.melt(polar, id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=600)
fig4.show()
