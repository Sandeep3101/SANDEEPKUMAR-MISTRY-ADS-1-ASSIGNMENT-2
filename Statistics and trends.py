# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:06:00 2022

@author: sanju
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

''' read_file is function for data reading World bank data in original dataframe and
    tranposed dataframe(dataFrame is showing country as column & t_dataFrame is 
    showing year as a column '''
    
def read_file(filename):
    
    dataFrame = pd.read_csv(filename)
    value_Name_indicator = dataFrame['Indicator Name'][0]
    indicator_code = dataFrame['Indicator Code'][0]
    dataFrame = dataFrame.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
    Years = list(dataFrame.columns)
    t_dataFrame = pd.melt(dataFrame, id_vars=['Country Name'], value_vars=Years, var_name='Years', value_name=value_Name_indicator)   

    return dataFrame, t_dataFrame


''' extract data of urban population '''

# read file by calling function
dataFrame, t_dataFrame = read_file("API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_4748769.csv")
print(dataFrame)
print(t_dataFrame)

# selecting rows by slicing method
urban_population = dataFrame.iloc[[20,29,35,40,77,81,109,174,184,251],:]
print(urban_population)

# drop Nan values
urban_population_drop = urban_population.dropna()
print(urban_population_drop)

# reseting index of selected data
urban_population_reindex = urban_population_drop.reset_index()
print(urban_population_reindex)

# choosing perticular columns of years
urban_population_countries = urban_population_reindex.iloc[:,[1,2,52,57,62]]
print(urban_population_countries)

# save a excel file
urban_population_countries.to_excel("urban.xlsx")


''' extract a data of forest area and plotting a line graph '''
 
# read file by calling function
dataFrame, t_dataFrame = read_file("API_AG.LND.FRST.ZS_DS2_en_csv_v2_4748391.csv")
print(dataFrame)
print(t_dataFrame)

# selecting data by slicing method and drop it
forest_area = dataFrame.drop(dataFrame.iloc[:,1:31], axis = 1) 
print(forest_area)

# drop Nan values
forest_area_drop = forest_area.dropna()
print(forest_area_drop)

# calculating average of total arable land through "numpy" module
print("\nAverage forest area: \n", forest_area_drop.mean())

# calculating Normal Distribution of particular year through "scipy" module
print("\nNormal Distribution: \n", stats.skew(forest_area_drop["2000"]))

# reseting index of selected data
forest_area_reindex = forest_area_drop.set_index('Country Name')
print(forest_area_reindex)

# transpose processed data
forest_area_trans = pd.DataFrame.transpose(forest_area_reindex)
print(forest_area_trans)

#setting figure size and dpi for better visualiation 
plt.figure(figsize=(10,8),dpi=720)

#plotting a line graph of extracted data 
plt.plot(forest_area_trans["Bangladesh"],linestyle = 'dashdot',label = "Bangladesh")
plt.plot(forest_area_trans["Brazil"],linestyle = 'dashdot',label = "Brazil")
plt.plot(forest_area_trans["Canada"],linestyle = 'dashdot',label = "Canada")
plt.plot(forest_area_trans["China"],linestyle = 'dashdot',label = "China")
plt.plot(forest_area_trans["France"],linestyle = 'dashdot',label = "France")
plt.plot(forest_area_trans["India"],linestyle = 'dashdot',label = "India")
plt.plot(forest_area_trans["Nigeria"],linestyle = 'dashdot',label = "Nigeria")
plt.plot(forest_area_trans["Pakistan"],linestyle = 'dashdot',label = "Pakistan")
plt.plot(forest_area_trans["United Kingdom"],linestyle = 'dashdot',label = "United Kingdom")
plt.plot(forest_area_trans["United States"],linestyle = 'dashdot',label = "United States")

#setting xticks, fontsize, title & labels
plt.xticks([0,5,10,15,20,25,30], fontsize = "14")
plt.legend(loc=(1.01,0.3), fontsize = "14")
plt.title("Forest Area", fontsize = "18")
plt.xlabel("Year", fontsize = "18")
plt.tight_layout()
plt.ylabel("")

#save a png figure
plt.savefig('forest.png')
plt.show()


''' extract a data of arable land and plotting a line graph '''

# read file by calling function
dataFrame, t_dataFrame = read_file("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_4749667.csv")
print(dataFrame)
print(t_dataFrame)

# reseting index of selected data
arable_land = dataFrame.set_index('Country Name')
print(arable_land)

# calculating average of total arable land through "numpy" module
print("\nAverage Arable Land: \n", arable_land.mean())

# transpose processed data
arable_land_trans = pd.DataFrame.transpose(arable_land)
print(arable_land_trans)

#setting figure size and dpi for better visualiation 
plt.figure(figsize=(10,8),dpi=720)

#plotting a line graph of extracted data 
plt.plot(arable_land_trans["Bangladesh"],linestyle = 'dashdot',label = "Bangladesh")
plt.plot(arable_land_trans["Brazil"],linestyle = 'dashdot',label = "Brazil")
plt.plot(arable_land_trans["Canada"],linestyle = 'dashdot',label = "Canada")
plt.plot(arable_land_trans["China"],linestyle = 'dashdot',label = "China")
plt.plot(arable_land_trans["France"],linestyle = 'dashdot',label = "France")
plt.plot(arable_land_trans["India"],linestyle = 'dashdot',label = "India")
plt.plot(arable_land_trans["Nigeria"],linestyle = 'dashdot',label = "Nigeria")
plt.plot(arable_land_trans["Pakistan"],linestyle = 'dashdot',label = "Pakistan")
plt.plot(arable_land_trans["United Kingdom"],linestyle = 'dashdot',label = "United Kingdom")
plt.plot(arable_land_trans["United States"],linestyle = 'dashdot',label = "United States")

#setting xticks, fontsize, title & labels
plt.xticks([10,20,30,40,50,], fontsize = "14")
plt.legend(loc=(1.01,0.3), fontsize = "14")
plt.title("Arable Land", fontsize = "18")
plt.xlabel("Year", fontsize = "18")
plt.tight_layout()

#save a png figure
plt.savefig('arable.png')
plt.show()


'''extract a data of greenhouse gas emission plotting a bar graph '''

# read file by calling function
dataFrame, t_dataFrame = read_file("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_4748555.csv")
print(dataFrame)
print(t_dataFrame)

# selecting data by slicing method
greenhouse_emission = dataFrame.iloc[[20,29,35,40,77,81,109,174,184,251],:]
print(greenhouse_emission)


# calculating Normal Distribution of particular year through "scipy" module
print("\nNormal Distribution: \n", stats.skew(greenhouse_emission["1990"]))

# reseting index of selected data
greenhouse_emission_reindex = greenhouse_emission.set_index('Country Name')
print(greenhouse_emission_reindex)

# Seleting country Columns From DataFrame
greenhouse_emission_s = (greenhouse_emission["Country Name"])

# Setting The Length Of country names
X_axis = np.arange(len(greenhouse_emission_s))

# arranging Columns From DataFrame For Plotting
_1990_ = (greenhouse_emission["1990"])
_1995_ = (greenhouse_emission["1995"])
_2000_ = (greenhouse_emission["2000"])
_2005_ = (greenhouse_emission["2005"])
_2010_ = (greenhouse_emission["2010"])
_2015_ = (greenhouse_emission["2015"])

# For Better Visualisation Set Fig configuration
plt.figure(figsize=(10,8), dpi=500)

# Plotting Bar Graph of Selected Columns
plt.bar(X_axis - 0.3, _1990_, 0.1, label="1990",edgecolor = "black")
plt.bar(X_axis - 0.2, _1995_, 0.1, label="1995",edgecolor = "black")
plt.bar(X_axis - 0.1, _2000_, 0.1, label="2000",edgecolor = "black")
plt.bar(X_axis + 0.0, _2005_, 0.1, label="2005",edgecolor = "black")
plt.bar(X_axis + 0.1, _2010_, 0.1, label="2010",edgecolor = "black")
plt.bar(X_axis + 0.2, _2015_, 0.1, label="2015",edgecolor = "black")

# Setting x & y labels, Showing the legend, Title And X axis Ticks
plt.title("Greenhouse Emission", fontsize = "18")
plt.xlabel("Country Name", fontsize = "18")
plt.xticks(X_axis, greenhouse_emission_s, rotation = 45, fontsize = "14")
plt.legend(fontsize = "14")
plt.tight_layout()

#save a png figure
plt.savefig("greenhouse emission.png")
plt.show()


''' extract a data of GDP plotting a bar graph '''

# read file by calling function
dataFrame, t_dataFrame = read_file("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4751183.csv")
print(dataFrame)
print(t_dataFrame)

# selecting data by slicing method
GDP = dataFrame.iloc[[20,29,35,40,77,81,109,174,184,251],:]
print(GDP)

# drop Nan values
GDP_drop = GDP.dropna()
print(GDP_drop)

# reseting index of selected data
GDP_countries = GDP_drop.set_index('Country Name')
print(GDP_countries)

# Seleting country Columns From DataFrame
GDP_s = (GDP_drop["Country Name"])

# calculating average of total GDP through "numpy" module
print("\nAverage GDP: \n", GDP_drop.mean())

# arranging The Length of country columns
X_axis = np.arange(len(GDP_s))

# Seleting Columns From DataFrame For Plotting
_1990_ = (GDP["1990"])
_1995_ = (GDP["1995"])
_2000_ = (GDP["2000"])
_2005_ = (GDP["2005"])
_2010_ = (GDP["2010"])
_2015_ = (GDP["2015"])

# For Better Visualisation Set Fig configuration
plt.figure(figsize=(10,8), dpi=500)

# Plotting Bar Graph of Selected Columns
plt.bar(X_axis - 0.3, _1990_, 0.1, label="1990",edgecolor = "black", color = "red")
plt.bar(X_axis - 0.2, _1995_, 0.1, label="1995",edgecolor = "black", color = "green" )
plt.bar(X_axis - 0.1, _2000_, 0.1, label="2000",edgecolor = "black", color = "brown")
plt.bar(X_axis + 0.0, _2005_, 0.1, label="2005",edgecolor = "black", color = "indigo")
plt.bar(X_axis + 0.1, _2010_, 0.1, label="2010",edgecolor = "black", color = "orange")
plt.bar(X_axis + 0.2, _2015_, 0.1, label="2015",edgecolor = "black", color = "grey")

# Setting x & y labels, Showing the legend, Title And X axis Ticks
plt.title("GDP", fontsize = "18")
plt.xlabel("Country Name", fontsize = "18")
plt.xticks(X_axis, GDP_s, rotation = 45, fontsize = "14")
plt.legend(fontsize = "14")
plt.tight_layout()

# save a png figure
plt.savefig("gdp2.png")
plt.show()


''' extract a data of China and India plotting a heatmap ''' 


#read excel file by pandas
countries = pd.read_excel("API_19_DS2_en_excel_v2_4700532.xls")
print(countries)

# selecting china country by slicing method
china_s= countries.iloc[3040:3116]
print(china_s)

# reseting index of selected data
china_reset = china_s.reset_index().set_index("Indicator Name")
print(china_reset)

# drop unwanted columns
china_drop = china_reset.drop(["index", "Country Name", "Country Code", "Indicator Code"], axis = 1)
print(china_drop)

# drop unwanted indicators
china_c= china_drop.iloc[[0,22,34,50,66,73],:]
print(china_c)

# transpose processed data
china_trans= pd.DataFrame.transpose(china_c)
print(china_trans)

# finding correlation of extracted data by numpy module
print("\correlation of indicatiors: \n", china_trans.corr())

# For Better Visualisation Set Fig configuration
plt.figure(figsize=(10,8),dpi=1040)

# plotting heatmap by extracted correlation data
sns.heatmap(china_trans.corr(), annot=True, vmin=-1.0, vmax=+1.0, cmap='RdBu')
plt.title("China", fontsize = "18")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()

# save a png figure
plt.savefig('china.png')
plt.show()

# selecting india country by slicing method
india_s= countries.iloc[8284:8359]
print(india_s)

# reseting index of selected data
india_reset = india_s.reset_index().set_index("Indicator Name")
print(india_reset)

# drop unwanted columns
india_drop = india_reset.drop(["index", "Country Name", "Country Code", "Indicator Code"], axis = 1)
print(india_drop)

# drop unwanted indicators
india_c= india_drop.iloc[[0,22,34,50,66,73],:]
print(india_c)

# transpose processed data
india_trans= pd.DataFrame.transpose(india_c)
print(india_trans)

# finding correlation of extracted data by numpy module
print("\correlation of indicatiors: \n", india_trans.corr())

# For Better Visualisation Set Fig configuration
plt.figure(figsize=(10,8),dpi=1080)

# plotting heatmap by extracted correlation data
sns.heatmap(india_trans.corr(), annot=True, vmin=-1.0, vmax=+1.0, cmap='coolwarm')
plt.title("India", fontsize = "18")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig('india.png')

# save a png figure
plt.show()
