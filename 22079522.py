# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:44:33 2024

@author: LENOVO
"""

"""
This program reads a data set , cleans and prepare the set for cluster analysis.
It can standardise,  normalise and fit regression on clusters. The data is 
drawn from World bank dataset. 

"""

# Loading the required modules
import wbgapi as wb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def ReadWorldBankData(*args):  # Function definition
    '''
Takes country as first parameter and multiple parameters for climate \
change indicators.Provide the id code from World bank data as second \
set of arguments
    '''
    Read_series_country = wb.data.DataFrame(
        *args)  # Getting data from Worldbank data set
    Read_series_country.index.name = None
    Read_series_country.transform(np.sort, axis=1)
    # Transpose of the downloaded data set
    Read_series_country_Transpose = Read_series_country.transpose()
    dfRead_series_country_Transpose = pd.DataFrame(
        Read_series_country_Transpose)
    return dfRead_series_country_Transpose


"""
This section loads all the data from World bank data set using functions
"""
WBData_Ele_HIC = ReadWorldBankData(
    'EG.USE.ELEC.KH.PC', wb.income.members('HIC'))
WBData_Ele_LIC = ReadWorldBankData(
    'EG.USE.ELEC.KH.PC', wb.income.members('LIC'))
WBDataP_GDP_HIC = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members('HIC'))
WBData_GDP_LIC = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members('LIC'))


"""
In the following section data is cleaned before setting the dataframe for analysis.
"""
"""
First the high income countries are cleaned (HIC)
"""


def Countryset_Cleaned(df):
    """
    This function cleans the dataframe along rows and columns by removing the 
    Null value cells. 

    """
    df.index.name = 'Year'  # Give a name Year to the index, since it is year column by default
    df_Clean_R = df.dropna(how='all', axis=0, inplace=False)
    df_Clean_C = df_Clean_R.dropna(how='all', axis=1, inplace=False)
    #removed all rows with null values.
    df_Clean_C = df_Clean_C.dropna(how='any', axis=0, inplace=False)
    # If there is any null values corresponding column is removed
    return df_Clean_C


"""
Calling the Countryset_cleaned to clean the downloaded dataset
It accepts a dataframe as argument and returns a dataframe.
"""

HIC_C = Countryset_Cleaned(WBData_Ele_HIC)
# Applied the Countryset_Cleanedfunction on the HIC data set. 


def TransposeDF(df):
    """
    This function traspose the given data set and rename
    the columns (eg:countries)
    """

    df.reset_index()
    df_T = df.T.rename_axis('Countries').rename_axis(
        columns=None).reset_index()
    df_T = df_T.set_index('Countries')

    return df_T


Transpose_1990 = TransposeDF(HIC_C)
# Applied the TransposeDF on the HIC dataset
HIC_1990_2014 = Transpose_1990.loc[:, ['YR1990', 'YR2014']]
"""
The dataset is now filtered down to Year1990-Year2014 avoiding Null values.
This step was required since auto filteration with 'all' or 'any' removes
almost all data from the dataset. 
"""
#===================================================

"""
In this section onwards we are starting the clustering process. To do this
the data set is intially standardised and then normalised. This will ensure
any data set with varying units and abnormal values are fitted in to the 
analytical frame.
"""

scaler = StandardScaler()
Temp_scaled = pd.DataFrame(scaler.fit_transform(HIC_1990_2014),\
                           columns=HIC_1990_2014.columns)
Temp_scaled = scaler.fit_transform(HIC_1990_2014)

"""
In the above step the data from High income countries between 1990 and 2014
are standardised and normalised, which returns an array structure. 
"""
# ===================================================
"""
Initialisation to calculate k-means, were the plotted graph provides an
elbow graph were the bend  is selected as the numbver of optimal clusters.  
"""
kmeans_init = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
}

# create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_init)
    kmeans.fit(Temp_scaled)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('Figure.1::Elbow plot for cluster identification')
plt.savefig('Number_Of_Clusters.png', bbox_inches="tight", dpi=300)
plt.clf()#The plot is cleared so that it does not overlap with next plot

# ===================================================

# Initialize the class object

kmeans = KMeans(n_clusters=5)

# predict the labels of clusters.
label = kmeans.fit_predict(Temp_scaled)
# ======================================================

# Regression plot for each cluster

data_with_labels = np.column_stack((Temp_scaled, label))

for cluster_label in range(5):
    cluster_data = data_with_labels[data_with_labels[:, -1]
                                    == cluster_label][:, :-1]

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(cluster_data[:, 0].reshape(-1, 1), cluster_data[:, 1])

    # Plot the data points
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                label=f'Cluster {cluster_label}')

    # Plot the regression line
    sns.regplot(x=cluster_data[:, 0], y=cluster_data[:, 1],
                scatter=False, ax=plt.gca(), color='red')

plt.title('Figure.3::Regression plot for each cluster')
plt.xlabel('YR1990')
plt.ylabel('YR2014')
plt.legend()
plt.savefig('Regression plot for each cluster.png',
            bbox_inches="tight", dpi=300)
plt.clf()#The plot is cleared so that it does not overlap with next plot
# ======================================================

# Getting unique labels

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

for i in u_labels:
   ax2 = plt.scatter(Temp_scaled[label == i, 0],
                Temp_scaled[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.title('Figure.2::Centroids plot')
plt.savefig('Plotting the centroids.png',
            bbox_inches="tight", dpi=300)
plt.clf()
# ============================================================================
"""
This section provides the plot of regression lines over the identified clusters.
Out of 5 clusters, only 3 were showing proper regression lines, 
since two clusters are formed by extreme values and have no significant
neighbouring values. 
"""
Temp_scaled = pd.DataFrame(scaler.fit_transform(
    HIC_1990_2014), columns=HIC_1990_2014.columns)
HIC_1990_2014['label'] = kmeans.fit_predict(Temp_scaled[['YR1990', 'YR2014']])

# Get centroids and inverse transform to de-normalise the normalised data.

centroids2 = scaler.inverse_transform(kmeans.cluster_centers_)
cen_x = [i[0] for i in centroids2]
cen_y = [i[1] for i in centroids2]

ax3 = sns.scatterplot(x='YR1990', y='YR2014', hue='label',data=HIC_1990_2014, palette='colorblind',
                     legend='full')

HIC_1990_2014 = HIC_1990_2014.reset_index()

#Plotting the original dataset. 

ax = sns.scatterplot(HIC_1990_2014, x="YR1990", y="YR2014",
                    size='YR2014', hue=HIC_1990_2014['label'], palette="seismic")

#Plotting the centroids on the selected clusters. 
sns.scatterplot(x=cen_x, y=cen_y, s=100, color='black', ax=ax)

# Regression plot on the identified clusters. 
sns.regplot(HIC_1990_2014, x="YR1990", y="YR2014", scatter_kws={
            "color": "black", "alpha": 0.3}, line_kws={"color": "red"}, ci=99)
ax.legend_.remove()

plt.title("Figure.4:: Plotting the regression line over the full data set with error band.") 

plt.tight_layout()
plt.savefig('Regression plot with error bands on original data set.png',
            bbox_inches="tight", dpi=300)
plt.clf()
