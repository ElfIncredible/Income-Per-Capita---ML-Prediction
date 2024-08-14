# Income Per Capita - ML Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#achine-learning-prediction)
  - [Install Dependencies](#install-dependencies)
  - [Data Collection and Processing](#data-collection-and-processing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)

## Project Overview

## Dataset

## Machine Learning Prediction
### Install Dependencies
Set up the necessary libraries for data handling, visualization, and machine learning, specifically for tasks related to training and evaluating a Random Forest regression model.

```
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```
### Data Collection and Processing
Prepare and clean a dataset by configuring display settings, loading the data, inspecting its structure, removing unnecessary columns, and handling missing values.
- Modify the display settings in pandas to show all rows and columns in the DataFrame when printing it, which can be useful for examining the data in its entirety.
  ```
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  ```
- Read the CSV file into a pandas DataFrame. The skiprows=4 argument skips the first four rows of the file, which are often headers or metadata, to correctly load the data.
  ```
  df = pd.read_csv('/kaggle/input/gdp-income-per-capita-2/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3152603.csv', skiprows=4)
  ```
- Display the first five rows of the DataFrame to give an overview of the data.
  ```
  df.head()
  ```
- List the column names in the DataFrame.
  ```
  df.columns
  ```
- Get a concise summary of the DataFrame, including the number of non-null entries and the data types of each column.
  ```
  df.info()
  ```
- Generate descriptive statistics for numerical columns, including count, mean, standard deviation, minimum, and maximum values.
  ```
  df.describe()
  ```
- Check the number of rows and columns in the DataFrame.
  ```
  Returns a tuple representing the number of rows and columns in the DataFrame.
  ```
- Calculates the number of missing (NaN) values in each column.
  ```
  Calculates the number of missing (NaN) values in each column.
  ```
- Remove specified columns from the DataFrame that are not needed for the analysis.
  ```
  df = df.drop(columns=['Unnamed: 68', 'Country Code', 'Indicator Name', 'Indicator Code'])
  ```
- Set the 'Country Name' column as the index of the DataFrame, which can be useful for accessing rows by country name.
  ```
  df = df.set_index('Country Name')
  ```
- Fill missing values in the DataFrame:
    - _ffill(axis=1) (forward fill)_ propagates the last valid observation forward to the next valid observation within each row.
      ```
      df = df.ffill(axis=1)
      ```
    - _bfill(axis=1) (backward fill)_ propagates the next valid observation backward to fill missing values. Together, these methods ensure that missing values are filled in each row using adjacent data.
      ```
      df = df.bfill(axis=1)
      ```
### Exploratory Data Analysis
1. Visualize the GDP per capita data for Kenya as a line plot, showing how it changes over the years.
```
# Plot GDP per capita for Kenya

# Ensure 'Country Name' is the index and use .loc to select Kenya's data
kenya_data = df.loc['Kenya']

# Extract years and GDP per capita values
years = kenya_data.index
gdp_per_capita = kenya_data.values

# Plotting
plt.figure(figsize=(20, 8))
plt.plot(years, gdp_per_capita, marker='o', linestyle='-', color='b')
plt.title('GDP Per Capita for Kenya')
plt.xlabel('Year')
plt.ylabel('GDP Per Capita')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/cead41d7-b6dd-45e9-aefb-e39677cd0069)
