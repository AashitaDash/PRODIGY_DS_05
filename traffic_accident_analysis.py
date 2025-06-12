import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Inter' # Set font to Inter

# --- 1. Simulate a Traffic Accident Dataset ---
# This simulation aims to mimic the structure and types of data
# found in real-world traffic accident datasets, making the code runnable
# without requiring a large external download.

num_accidents = 5000 # Number of simulated accident records

# Simulate geographical locations (simplified for demonstration)
simulated_cities = {
    'Los Angeles, CA': (34.0522, -118.2437),
    'New York, NY': (40.7128, -74.0060),
    'Chicago, IL': (41.8781, -87.6298),
    'Houston, TX': (29.7604, -95.3698),
    'Phoenix, AZ': (33.4484, -112.0740),
    'Miami, FL': (25.7617, -80.1918),
    'Seattle, WA': (47.6062, -122.3321)
}
city_names = list(simulated_cities.keys())

data = {
    'ID': [f'A-{i+1}' for i in range(num_accidents)],
    'Severity': np.random.randint(1, 5, num_accidents), # 1: Minor, 4: Severe
    'Start_Time': [datetime(2023, 1, 1, 0, 0, 0) + timedelta(minutes=random.randint(0, 365*24*60)) for _ in range(num_accidents)],
    'End_Time': [None] * num_accidents, # Simplification: End_Time not used for analysis
    'Latitude': [0.0] * num_accidents,
    'Longitude': [0.0] * num_accidents,
    'City': np.random.choice(city_names, num_accidents),
    'State': [city.split(', ')[1] for city in np.random.choice(city_names, num_accidents)], # Will be corrected below
    'Temperature(F)': np.random.normal(60, 20, num_accidents).round(1),
    'Humidity(%)': np.random.normal(70, 15, num_accidents).round(1),
    'Pressure(in)': np.random.normal(29.9, 0.5, num_accidents).round(2),
    'Visibility(mi)': np.random.normal(9, 2, num_accidents).clip(0, 10).round(1), # Clip to realistic visibility
    'Weather_Condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow', 'Heavy Rain', 'Thunderstorm', 'Windy'], num_accidents, p=[0.4, 0.2, 0.15, 0.08, 0.05, 0.05, 0.04, 0.03]),
    'Road_Condition': np.random.choice(['Dry', 'Wet', 'Ice', 'Snow', 'Slippery', 'Flooded'], num_accidents, p=[0.6, 0.25, 0.05, 0.05, 0.03, 0.02]),
    'Wind_Speed(mph)': np.random.normal(10, 5, num_accidents).clip(0, 50).round(1),
    'Amenity': np.random.randint(0, 2, num_accidents), # 0 or 1
    'Traffic_Signal': np.random.randint(0, 2, num_accidents),
    'Junction': np.random.randint(0, 2, num_accidents),
    'Sunrise_Sunset': np.random.choice(['Day', 'Night'], num_accidents),
    'Civil_Twilight': np.random.choice(['Day', 'Night'], num_accidents)
}

df = pd.DataFrame(data)

# Assign Latitude/Longitude based on the chosen city
for i, row in df.iterrows():
    lat, lon = simulated_cities[row['City']]
    df.at[i, 'Latitude'] = lat + np.random.uniform(-0.5, 0.5) # Add some noise
    df.at[i, 'Longitude'] = lon + np.random.uniform(-0.5, 0.5) # Add some noise
    df.at[i, 'State'] = row['City'].split(', ')[1] # Correct state based on city

# Introduce some missing values for demonstration of handling them
missing_indices_temp = np.random.choice(df.index, 50, replace=False)
df.loc[missing_indices_temp, 'Temperature(F)'] = np.nan
missing_indices_wc = np.random.choice(df.index, 20, replace=False)
df.loc[missing_indices_wc, 'Weather_Condition'] = np.nan

print("--- Simulated Traffic Accident Data Created ---")
print(df.head())
print(f"\nTotal accidents simulated: {len(df)}")
print("\nDataset Info (before detailed preprocessing):")
df.info()
print("\n")

# --- 2. Initial Data Cleaning & Feature Engineering ---

print("--- Data Cleaning & Feature Engineering ---")

# Convert Start_Time to datetime objects and extract time-based features
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Day_of_Week'] = df['Start_Time'].dt.day_name()
df['Hour'] = df['Start_Time'].dt.hour
df['Minute'] = df['Start_Time'].dt.minute

print("Extracted Year, Month, Day_of_Week, Hour, Minute from Start_Time.")

# Handle missing values: Impute numerical with median, categorical with mode
for col in ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Missing '{col}' values filled with median: {median_val}")

for col in ['Weather_Condition', 'Road_Condition', 'Sunrise_Sunset', 'Civil_Twilight']:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Missing '{col}' values filled with mode: {mode_val}")

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())
print("\n")

# --- 3. Exploratory Data Analysis (EDA) & Visualization ---

print("--- Exploratory Data Analysis & Visualization ---")

# 1. Accidents by Hour of Day
plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', data=df, palette='viridis', edgecolor='black')
plt.title('Number of Accidents by Hour of Day', fontsize=18, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Accidents by Day of Week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_Week', data=df, palette='plasma', order=day_order, edgecolor='black')
plt.title('Number of Accidents by Day of Week', fontsize=18, fontweight='bold')
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Accidents by Month
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['Month_Name'] = df['Start_Time'].dt.month_name()
plt.figure(figsize=(12, 6))
sns.countplot(x='Month_Name', data=df, palette='coolwarm', order=month_order, edgecolor='black')
plt.title('Number of Accidents by Month', fontsize=18, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Accidents by Weather Condition (Top N conditions)
top_weather_conditions = df['Weather_Condition'].value_counts().head(10).index
plt.figure(figsize=(14, 7))
sns.countplot(y='Weather_Condition', data=df[df['Weather_Condition'].isin(top_weather_conditions)],
              order=top_weather_conditions, palette='viridis', edgecolor='black')
plt.title('Top 10 Weather Conditions Causing Accidents', fontsize=18, fontweight='bold')
plt.xlabel('Number of Accidents', fontsize=14)
plt.ylabel('Weather Condition', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Accidents by Road Condition
top_road_conditions = df['Road_Condition'].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.countplot(x='Road_Condition', data=df[df['Road_Condition'].isin(top_road_conditions)],
              order=top_road_conditions, palette='magma', edgecolor='black')
plt.title('Accidents by Road Condition', fontsize=18, fontweight='bold')
plt.xlabel('Road Condition', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. Accident Hotspots (Top Cities by Accident Count)
top_cities = df['City'].value_counts().head(10).index
plt.figure(figsize=(12, 7))
sns.countplot(y='City', data=df[df['City'].isin(top_cities)], order=top_cities, palette='rocket', edgecolor='black')
plt.title('Top 10 Cities by Accident Count (Simulated Hotspots)', fontsize=18, fontweight='bold')
plt.xlabel('Number of Accidents', fontsize=14)
plt.ylabel('City', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 7. Severity vs. Temperature (Example of Factor Impact)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity', y='Temperature(F)', data=df, palette='coolwarm')
plt.title('Accident Severity by Temperature (F)', fontsize=18, fontweight='bold')
plt.xlabel('Severity (1: Minor, 4: Severe)', fontsize=14)
plt.ylabel('Temperature (F)', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 8. Accident Hotspots (Geographical Scatter Plot - Simplified)
# This is a basic scatter plot to represent geographical distribution.
# For actual 'hotspots', density plots or heatmaps on real maps (e.g., Folium) are better.
plt.figure(figsize=(12, 10))
sns.scatterplot(x='Longitude', y='Latitude', hue='Severity', size='Severity',
                sizes=(20, 400), alpha=0.6, palette='viridis', data=df,
                legend='full', edgecolor='w', linewidth=0.5)
plt.title('Simulated Accident Locations by Severity', fontsize=18, fontweight='bold')
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


print("\n--- Traffic Accident Data Analysis Complete ---")
print("Data simulated, preprocessed, and visualizations displayed for patterns related to time, weather, road conditions, and geographical distribution.")
