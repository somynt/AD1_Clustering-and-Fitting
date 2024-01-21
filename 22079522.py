# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:44:33 2024

@author: LENOVO
"""

import wbgapi as wb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn import cluster 
#Loading the required modules
 
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def ReadWorldBankData (*args):  # Function definition
    '''
Takes country as first parameter and multiple parameters for climate \
change indicators.Provide the id code from World bank data as second \
set of arguments
    '''
    Read_series_country = wb.data.DataFrame(*args)  # Getting data from Worldbank data set
    Read_series_country.index.name = None
    Read_series_country.transform(np.sort, axis=1)
    #Transpose of the downloaded data set
    Read_series_country_Transpose = Read_series_country.transpose()
    dfRead_series_country_Transpose = pd.DataFrame(Read_series_country_Transpose)
    return dfRead_series_country_Transpose


"""
This section loads all the data from World bank data set using functions
"""
#WBDataPop_GDP = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members(
   # 'HIC'))  # Load World bank data for GDP in high Income Countries (HIC)
#WBDataPop_Transpose = ReadWorldBankData('SP.POP.TOTL', wb.income.members('HIC'))  # World bank data population of high income countries
# Load World bank data for World GDP
#WBDataPop_WorldGDP = ReadWorldBankData('NY.GDP.PCAP.CD')
# World bank data for World Population
#WBData_World_Pop = ReadWorldBankData('SP.POP.TOTL')

WBData_Ele_HIC = ReadWorldBankData('EG.USE.ELEC.KH.PC', wb.income.members('HIC'))
WBData_Ele_LIC = ReadWorldBankData('EG.USE.ELEC.KH.PC', wb.income.members('LIC'))
WBDataP_GDP_HIC = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members('HIC'))
WBData_GDP_LIC = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members('LIC'))


"""
In the following section data is cleaned before setting the dataframe for analysis.
"""
"""
First the high income countries are cleaned (HIC)
"""

def Countryset_Cleaned (df):
     
    """
    This function cleans the dataframe along rows and columns

    """
    df.index.name ='Year' # Give a name Year to the index, since it is year column by default
    df_Clean_R = df.dropna(how='all', axis=0, inplace=False)
    df_Clean_C = df_Clean_R.dropna(how='all', axis=1, inplace=False)
    df_Clean_C = df_Clean_C.dropna(how='any', axis=0, inplace=False)
    
    return df_Clean_C

"""
Calling the Countryset_cleaned to clean the downloaded dataset
It accepts a dataframe as argument and returns a dataframe.
"""

HIC_C = Countryset_Cleaned(WBData_Ele_HIC)
LIC_C = Countryset_Cleaned(WBData_Ele_HIC)

"""
HIC_C = HIC_C.reset_index()
HIC_C['Year'] = HIC_C['Year'].str.extract(r'YR(\d{4})')
HIC_C ['Year'] = pd.to_datetime(HIC_C ['Year'])
HIC_C ['Year'] = HIC_C ['Year'].dt.year
HIC_C.set_index('Year', inplace=True)
"""

#HIC_GDP_C = Countryset_Cleaned(WBDataP_GDP_HIC)
#LIC_GDP_C = Countryset_Cleaned(WBData_GDP_LIC)

#Combined_df = pd.merge(HIC_C, LIC_C, on='Year') # Merging similar type datasets with temperature
#Combined_df_GDP = pd.merge(HIC_GDP_C, LIC_GDP_C, on = 'Year')# Merging similar type datasets with GDP

def TransposeDF (df):
  """
    This function traspose the given data set and rename the columns (eg:countries)
  """
    
  df.reset_index()
  df_T= df.T.rename_axis('Countries').rename_axis(columns =None).reset_index()
  df_T = df_T.set_index('Countries')

  return df_T

Transpose_1990 = TransposeDF(HIC_C)
#Transpose_1990  = Transpose_1990.reset_index()


HIC_1990_2014 = Transpose_1990.loc[:,['YR1990', 'YR2014']]
#HIC_1990_2014['Countries'] = HIC_1990_2014['Countries'].astype('string')


"""
S = sns.scatterplot (x = 'YR1990', y = 'YR2014' , hue = 'Countries', data = HIC_1990_2014)
S.legend_.remove()



#Transpose_1990['Countrycode'] = Transpose_1990.index
#print(HIC_1990_2014)


#scaled_df = StandardScaler().fit_transform(HIC_1990_2014)



#===============================================================


#HIC_1990_2014= HIC_1990_2014.set_index('Countries')
"""
scaler = StandardScaler()
#Temp_scaled = pd.DataFrame(scaler.fit_transform(HIC_1990_2014), columns=HIC_1990_2014.columns)
Temp_scaled = scaler.fit_transform(HIC_1990_2014)


"""
kmeans_init = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_init)
    kmeans.fit(Temp_scaled)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#===================================================
"""
#ORIGINAL


#Initialize the class object

kmeans = KMeans(n_clusters= 5)
 
#predict the labels of clusters.
label = kmeans.fit_predict(Temp_scaled)


#Getting unique labels

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(Temp_scaled[label == i , 0] , Temp_scaled[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()
#============================================================================

Temp_scaled = pd.DataFrame(scaler.fit_transform(HIC_1990_2014), columns=HIC_1990_2014.columns)
HIC_1990_2014['label'] = kmeans.fit_predict(Temp_scaled[['YR1990', 'YR2014']])

# get centroids
centroids2 = scaler.inverse_transform(kmeans.cluster_centers_)
cen_x = [i[0] for i in centroids2]
cen_y = [i[1] for i in centroids2]

ax = sns.scatterplot(x='YR1990', y='YR2014', hue='label',
                     data=HIC_1990_2014, palette='colorblind',
                     legend='full')


HIC_1990_2014 = HIC_1990_2014.reset_index()
ax = sns.scatterplot(HIC_1990_2014, x="YR1990", y="YR2014", size='YR2014', hue= HIC_1990_2014['label'], palette="seismic")

sns.scatterplot(x=cen_x, y=cen_y, s=100, color='black', marker ='x', ax=ax)
ax.legend_.remove()
plt.tight_layout()
plt.show()

