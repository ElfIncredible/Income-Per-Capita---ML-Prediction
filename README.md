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

2. Generate a line plot comparing the GDP per capita trends for Kenya and Singapore over time.
   ```
    # Filter the data for Kenya and Singapore
    kenya_data = df.loc['Kenya']
    singapore_data = df.loc['Singapore']

    # Extract years and GDP per capita values
    years = kenya_data.index
    gdp_per_capita_kenya = kenya_data.values
    gdp_per_capita_singapore = singapore_data.values

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(years, gdp_per_capita_kenya, marker='o', linestyle='-', color='b', label='Kenya')
    plt.plot(years, gdp_per_capita_singapore, marker='o', linestyle='-', color='r', label='Singapore')
    plt.title('GDP Per Capita Comparison: Kenya vs Singapore')
    plt.xlabel('Year')
    plt.ylabel('GDP Per Capita')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    ```

    ![image](https://github.com/user-attachments/assets/6541f4d0-bdca-4e4c-8f28-e82e5b3cdb01)

    - In 1968, Kenya and Singapore had the same GDP per capita, reflecting similar economic conditions at that time. However, since then, Singapore has experienced extraordinary economic growth, leading to a GDP per capita that is now 40 times higher than Kenya's. This dramatic disparity could be attributed to Singapore's successful economic policies, strategic investments in technology and education, robust infrastructure development, and effective governance, which propelled its rapid economic advancement. In contrast, Kenya has faced various challenges, including political instability, slower economic reforms, and infrastructure limitations, which have impacted its economic growth trajectory.

3.  Generate a line plot comparing the GDP per capita trends for Kenya and Malaysia, allowing for a visual comparison of their economic trajectories over the years.
    ```
    # Filter the data for Kenya and Malaysia
    kenya_data = df.loc['Kenya']
    malaysia_data = df.loc['Malaysia']

    # Extract years and GDP per capita values
    years = kenya_data.index
    gdp_per_capita_kenya = kenya_data.values
    gdp_per_capita_malaysia = malaysia_data.values

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(years, gdp_per_capita_kenya, marker='o', linestyle='-', color='b', label='Kenya')
    plt.plot(years, gdp_per_capita_malaysia, marker='o', linestyle='-', color='r', label='Malaysia')
    plt.title('GDP Per Capita Comparison: Kenya vs Malaysia')
    plt.xlabel('Year')
    plt.ylabel('GDP Per Capita')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    ```

    ![image](https://github.com/user-attachments/assets/42140045-a12f-4d76-8ff8-2b38545fb7be)

    - Up until 1972, Kenya and Malaysia had similar GDP per capita levels. Since then, Malaysia has surged ahead, with its GDP per capita now nearly six times higher than Kenya's. This disparity likely results from Malaysia's strategic economic policies, including industrialization, investment in education, and infrastructure development, which fueled its rapid economic growth. In contrast, Kenya has faced challenges such as political instability and slower economic reforms, which have impacted its growth trajectory.

4. Create a comparative line plot to visualize and analyze the GDP per capita trends for Kenya and South Africa, providing insights into how the economic performance of these two countries has evolved over time.
   ```
   # Filter the data for Kenya and South Africa
    kenya_data = df.loc['Kenya']
    south_africa_data = df.loc['South Africa']

    # Extract years and GDP per capita values
    years = kenya_data.index
    gdp_per_capita_kenya = kenya_data.values
    gdp_per_capita_south_africa = south_africa_data.values

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(years, gdp_per_capita_kenya, marker='o', linestyle='-', color='b', label='Kenya')
    plt.plot(years, gdp_per_capita_south_africa, marker='o', linestyle='-', color='r', label='South Africa')
    plt.title('GDP Per Capita Comparison: Kenya vs South Africa')
    plt.xlabel('Year')
    plt.ylabel('GDP Per Capita')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
   ```

   ![image](https://github.com/user-attachments/assets/a42e6f36-5c73-4d81-a342-2830402279ea)

   - In 1972, Kenya and South Africa had similar GDP per capita levels. Today, South Africa’s GDP per capita is nearly three times higher than Kenya’s. This disparity is due to South Africa’s effective economic policies, significant infrastructure investments, and economic diversification. In contrast, Kenya has faced challenges such as political instability, slower economic reforms, and reliance on agriculture, impacting its economic growth. Despite being in the same continent, these factors have led to divergent economic trajectories for the two countries.

  5. Create a comparative line plot to visualize and analyze the GDP per capita trends for Kenya and Nigeria, helping to understand the economic performance of both countries over time.
     ```
     # Filter the data for Kenya and Nigeria
    kenya_data = df.loc['Kenya']
    nigeria_data = df.loc['Nigeria']

    # Extract years and GDP per capita values
    years = kenya_data.index
    gdp_per_capita_kenya = kenya_data.values
    gdp_per_capita_nigeria = nigeria_data.values

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(years, gdp_per_capita_kenya, marker='o', linestyle='-', color='b', label='Kenya')
    plt.plot(years, gdp_per_capita_nigeria, marker='o', linestyle='-', color='r', label='Nigeria')
    plt.title('GDP Per Capita Comparison: Kenya vs Nigeria')
    plt.xlabel('Year')
    plt.ylabel('GDP Per Capita')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    ```

  ![image](https://github.com/user-attachments/assets/6c68d2ab-580e-42c7-9607-bbe793add48a)

6. Generate a line plot comparing the GDP per capita trends for Kenya and Uganda, providing a visual representation of their economic performance over time.
   ```
   # Filter the data for Kenya and Uganda
    kenya_data = df.loc['Kenya']
    uganda_data = df.loc['Uganda']

    # Extract years and GDP per capita values
    years = kenya_data.index
    gdp_per_capita_kenya = kenya_data.values
    gdp_per_capita_uganda = uganda_data.values

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(years, gdp_per_capita_kenya, marker='o', linestyle='-', color='b', label='Kenya')
    plt.plot(years, gdp_per_capita_uganda, marker='o', linestyle='-', color='r', label='Uganda')
    plt.title('GDP Per Capita Comparison: Kenya vs Uganda')
    plt.xlabel('Year')
    plt.ylabel('GDP Per Capita')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
   ```

   ![image](https://github.com/user-attachments/assets/3f65228b-e419-408d-b6a6-33d09f7e10e6)

7. 

   
