# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:40:11 2024

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


def ReadWorldBankData (*args):  # Function definition
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
#WBDataPop_GDP = ReadWorldBankData('NY.GDP.PCAP.CD', wb.income.members(
   # 'HIC'))  # Load World bank data for GDP in high Income Countries (HIC)
WBDataPop_Transpose = ReadWorldBankData('SP.POP.TOTL', wb.income.members(
    'HIC'))  # World bank data population of high income countries
# Load World bank data for World GDP
#WBDataPop_WorldGDP = ReadWorldBankData('NY.GDP.PCAP.CD')
# World bank data for World Population
#WBData_World_Pop = ReadWorldBankData('SP.POP.TOTL')

"""
In the following section data is cleaned before setting the dataframe for analysis.
"""

WBDataPop_Transpose.index.name = 'Year'

I = WBDataPop_Transpose #Indexed with name 'Year'
Year_DF_ResetIndex = WBDataPop_Transpose.reset_index() # Index reset to use the Year column
Countries_DF= Year_DF_ResetIndex.set_index('Year').T.rename_axis('Countries').rename_axis(None,axis=1) # Year_DF transposed to create \

""" 
new dataframe country_Df. This replaces the year in index and now country is the index.
"""
Countries_DF_ResetIndex = Countries_DF.reset_index() # Reset the 

Countries_DF ['Mean'] = Countries_DF.mean(axis=1)
Countries_DF_Sorted = Countries_DF.sort_values(by='Mean', ascending=False)
"""
row_labels = Countries_DF_Sorted.index.tolist()
print(row_labels)
"""
Countries_DF_New = Countries_DF_Sorted.filter(['USA', 'JPN', 'DEU', 'GBR', 'FRA', 'ITA', 'KOR', 'ESP', 'POL', 'CAN'], axis=0)
Countries_DF_New = Countries_DF_New.drop('Mean', axis=1)

"""
Drop both row and columns based on missing values. 
This returned same number of columns since there is no missing values 
"""
Countries_DF_New_Clean = Countries_DF_New.dropna(how='any', axis=0, inplace=False)





"""

Clustering Preparation: Initially scale each value to mean 0 and standard deviation 1

"""



print(Countries_DF_New_Clean.head())
"""
WBDataPop_Transpose_sorted = (WBDataPop_Transpose.sort_values(by=['YR1960', 'YR1961', 'YR1962', 'YR1963', 'YR1964', 'YR1965', 'YR1966',
                                                                  'YR1967', 'YR1968', 'YR1969', 'YR1970', 'YR1971', 'YR1972', 'YR1973',
                                                                  'YR1974', 'YR1975', 'YR1976', 'YR1977', 'YR1978', 'YR1979', 'YR1980',
                                                                  'YR1981', 'YR1982', 'YR1983', 'YR1984', 'YR1985', 'YR1986', 'YR1987',
                                                                  'YR1988', 'YR1989', 'YR1990', 'YR1991', 'YR1992', 'YR1993', 'YR1994',
                                                                  'YR1995', 'YR1996', 'YR1997', 'YR1998', 'YR1999', 'YR2000','YR2001',
                                                                  'YR2002', 'YR2003', 'YR2004', 'YR2005', 'YR2006', 'YR2007', 
                                                                  'YR2008', 'YR2009', 'YR2010', 'YR2011', 'YR2012', 'YR2013', 'YR2014', 
                                                                  'YR2015', 'YR2016', 'YR2017',
                                                                'YR2018', 'YR2019'], ascending=False, axis=1))

"""




