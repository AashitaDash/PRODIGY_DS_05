# PRODIGY_DS_05

Traffic Accident Data Analysis & Visualization
This repository contains Python code for analyzing and visualizing traffic accident data to identify patterns related to road conditions, weather, and time of day. The project also focuses on visualizing accident hotspots and contributing factors.

Project Goal
The primary objective of this project is to gain insights into factors contributing to traffic accidents, identify high-risk areas (hotspots), and understand temporal and environmental patterns. This analysis can help in informing road safety measures and policy decisions.

Project Structure
.
├── traffic_accident_analysis.py  # Main Python script for the analysis
└── README.md

Getting Started
Prerequisites
To run this code, you'll need Python installed along with the following libraries:

pandas

numpy

matplotlib

seaborn

You can install them using pip:

pip install pandas numpy matplotlib seaborn

Dataset
The code is designed to be runnable out-of-the-box by simulating a dataset that mimics the characteristics of real-world traffic accident data (e.g., US Accident dataset). This means you don't need to download any external files to run the provided script.

If you wish to work with a real dataset for further exploration, you can refer to datasets like the "US Accidents" dataset on Kaggle:

US Accidents Data on Kaggle

If you download such a dataset, you would modify the traffic_accident_analysis.py script to load this file instead of generating a simulated one.

Running the Code
Save the provided Python code (from the traffic_accident_analysis.py section below) into a file named traffic_accident_analysis.py.

Open your terminal or command prompt.

Navigate to the directory where you saved the file.

Execute the script using Python:

python traffic_accident_analysis.py

The script will print various outputs to the console, including data information and cleaning steps. It will then display several plots one by one. Close each plot window to proceed to the next one.

Code Overview (traffic_accident_analysis.py)
The traffic_accident_analysis.py script performs the following key steps:

Dataset Simulation: Generates a synthetic dataset with various features such as Severity, Start_Time, Latitude, Longitude, City, State, Temperature, Weather_Condition, Road_Condition, and more.

Initial Data Cleaning & Feature Engineering:

Converts Start_Time to datetime objects.

Extracts temporal features like Year, Month, Day_of_Week, Hour, and Minute.

Handles simulated missing values by imputing numerical columns with their median and categorical columns with their mode.

Exploratory Data Analysis (EDA) & Visualization: Generates a series of plots to visualize accident patterns and contributing factors:

Temporal Patterns: Number of accidents by Hour of Day, Day of Week, and Month.

Environmental Factors: Accident counts by Weather Condition and Road Condition.

Geographical Hotspots: Identification of top cities by accident count and a scatter plot of simulated accident locations colored by severity.

Factor Impact: Box plot showing Accident Severity by Temperature.

Visualizations
The script generates multiple plots to provide a comprehensive understanding of the accident data:

Histograms/Count Plots: Show distributions and counts of accidents across different categories (time, weather, road conditions).

Bar Plots: Illustrate the frequency of accidents in top cities and by specific conditions.

Box Plots: Help visualize the relationship between numerical factors (e.g., Temperature) and accident severity.

Scatter Plots: Provide a geographical representation of accidents, indicating simulated hotspots and severity.

Contributing
Feel free to fork this repository, open issues, or submit pull requests. Suggestions for improving data simulation, analysis techniques, or visualizations are highly welcome!

License
This project is open source and available under the MIT License.
