# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:18:22 2024

@author: LENOVO
"""

"""
The aim of this program is to analyse the electricity usage per capits (KWH)
for top 5 members of High Icome Group (HIC) and Low Income Group(LIC) from World Bank dataset.
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
#HIC_GDP_C = Countryset_Cleaned(WBDataP_GDP_HIC)
#LIC_GDP_C = Countryset_Cleaned(WBData_GDP_LIC)

#Combined_df = pd.merge(HIC_C, LIC_C, on='Year') # Merging similar type datasets with temperature
#Combined_df_GDP = pd.merge(HIC_GDP_C, LIC_GDP_C, on = 'Year')# Merging similar type datasets with GDP

def TransposeDF (df):
    """
    This function traspose the given data set and rename the columns (eg:countries)
    It accepts a dataframe as argument and returns a dataframe.
    """
    
    df.reset_index()
    df_T= df.T.rename_axis('Countries').rename_axis(columns =None).reset_index()
    df_T = df_T.set_index('Countries')

    return df_T

rr = TransposeDF(HIC_C)



#Combined_df_T = TransposeDF (Combined_df)
#Combined_df_T['AverageElecperCapita'] = round(Combined_df_T.mean(axis=1),2)
#Combined_df_T_Average = Combined_df_T.drop(Combined_df_T.loc[:,"YR1990":"YR2014"], axis=1)

#

#sns.lmplot(x = HIC_C.index, y = 'AverageElecperCapita', data = HIC_C)
#print (RI.head())

#sns.lmplot(x = 'Countries', y = 'Year', markers = ["s", "x"], palette = "Set2",data = Combined_df)

#Combined_df_GDP_T = TransposeDF(Combined_df_GDP)
#Combined_df_GDP_T_R= Combined_df_GDP_T.dropna(how='all', axis=1, inplace=False)
#Combined_df_GDP_T= Combined_df_GDP_T_R.dropna(how='any', axis=0, inplace=False)

#Combined_df_T['AverageElecperCapita'] = Combined_df_T.mean(axis=1)
#Combined_df_GDP_T['AverageGDP'] = Combined_df_GDP_T.mean(axis=1)


#Combined_df_T_Average = Combined_df_T.drop(Combined_df_T.loc[:,"YR1960":"YR2014"], axis=1)
#Combined_df_GDP_T_Average = Combined_df_GDP_T.drop(Combined_df_GDP_T.loc[:,"YR1960":"YR2022"], axis=1)

#Combine_Average_both = pd.merge(Combined_df_T_Average,Combined_df_GDP_T_Average, on= 'Countries' )

#print(Combined_df_T_Average.head())
#print(Combined_df_GDP_T_Average.head())

#print(Combined_df_T)

scaled_df = StandardScaler().fit_transform(rr)

"""
#===========================
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
#========================================
"""
"""
"""

#ORIGINAL


#Initialize the class object

kmeans = KMeans(n_clusters= 4)
 
#predict the labels of clusters.
label = kmeans.fit_predict(scaled_df)


#Getting unique labels

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

def create_dataframe(centroids):
    """
    Create a DataFrame from centroids.

    Parameters:
    - centroids: 2D NumPy array representing the centroids.

    Returns:
    - DataFrame with columns 'C1', 'C2', 'C3', 'C4' etc..
    """
    columns = [f'C{i+1}' for i in range(centroids.shape[1])]
    centroids_df = pd.DataFrame({col: centroids[:, i] for i, col in enumerate(columns)})
    return centroids_df

centroidsdf = create_dataframe(centroids)
scaled_df_df = create_dataframe (scaled_df)
#labeldf = create_dataframe (label)
labeldf = pd.DataFrame(data=label, columns = ['Clusters'])
print(labeldf)
for i in u_labels:
    plt.scatter(scaled_df[label == i , 0] , scaled_df[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')

plt.legend()
plt.show()


#Combined_df['Average'] = Combined_df.mean(axis=1)
#print(Combined_df.head())

