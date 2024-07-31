# To further refine this analysis and prepare for machine learning tasks to predict lithium demand, consider the following next steps:
# 1.Time Series Analysis:
# Implement ARIMA or Prophet models to forecast future battery replacements.
# Use the monthly_replacements data for this purpose.
# 2.Feature Engineering:
# Create additional features like rolling averages of replacements, seasonal indicators, etc.
# 3.Machine Learning Models:
# Implement Random Forest or Gradient Boosting models to predict replacements based on various features.
# Use cross-validation to ensure model robustness.
# 4.Scenario Analysis:
# Create different scenarios (e.g., rapid EV adoption, slow adoption) and predict lithium demand for each.
# 5.Model Evaluation:
# Implement functions to calculate RMSE, MAE, and MAPE for your predictive models.
# 6.Visualization Improvements:
# Consider using interactive plots (e.g., with plotly) for better exploration of the data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    date_columns = ['Start_Date', 'End_Date', 'Replacement_Date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])
    
    data['Usage_Duration'] = (data['Replacement_Date'] - data['Start_Date']).dt.days
    data['Avg_Daily_Distance'] = data['Total_Distance'] / data['Usage_Duration']
    data['Avg_Daily_Energy'] = data['Total_Energy_Consumed'] / data['Usage_Duration']
    
    data['Replacement_Year'] = data['Replacement_Date'].dt.year
    data['Start_Year'] = data['Start_Date'].dt.year
    
    return data

def plot_commuter_types(data):
    plt.figure(figsize=(10, 6))
    commuter_counts = data['Commuter_Type'].value_counts()
    sns.barplot(x=commuter_counts.index, y=commuter_counts.values)
    plt.title('Distribution of Commuter Types')
    plt.xlabel('Commuter Type')
    plt.ylabel('Count')
    plt.show()

def plot_usage_duration(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Usage_Duration'], bins=20, kde=True)
    plt.title('Distribution of Usage Duration')
    plt.xlabel('Usage Duration (days)')
    plt.ylabel('Frequency')
    plt.show()

def plot_daily_distance_vs_energy(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Avg_Daily_Distance', y='Avg_Daily_Energy', hue='Commuter_Type', data=data)
    plt.title('Average Daily Distance vs Average Daily Energy Consumption')
    plt.xlabel('Average Daily Distance (km)')
    plt.ylabel('Average Daily Energy Consumption (kWh)')
    plt.legend(title='Commuter Type')
    plt.show()

def plot_total_distance_by_commuter(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Commuter_Type', y='Total_Distance', data=data)
    plt.title('Total Distance by Commuter Type')
    plt.xlabel('Commuter Type')
    plt.ylabel('Total Distance (km)')
    plt.show()

def plot_usage_duration_vs_energy(data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Usage_Duration', y='Total_Energy_Consumed', hue='Commuter_Type', data=data)
    plt.title('Usage Duration vs Total Energy Consumed')
    plt.xlabel('Usage Duration (days)')
    plt.ylabel('Total Energy Consumed (kWh)')
    plt.legend(title='Commuter Type')
    plt.show()

def calculate_lithium_demand(data, lithium_per_battery):
    replacements_per_year = data['Replacement_Year'].value_counts().sort_index().reset_index()
    replacements_per_year.columns = ['Year', 'Number_of_Replacements']
    replacements_per_year['Lithium_Required_kg'] = replacements_per_year['Number_of_Replacements'] * lithium_per_battery

    new_ev_per_year = data['Start_Year'].value_counts().sort_index().reset_index()
    new_ev_per_year.columns = ['Year', 'Number_of_New_EVs']
    new_ev_per_year['Initial_Lithium_Required_kg'] = new_ev_per_year['Number_of_New_EVs'] * lithium_per_battery

    total_lithium_demand = pd.merge(new_ev_per_year, replacements_per_year, on='Year', how='outer').fillna(0)
    total_lithium_demand['Total_Lithium_Required_kg'] = total_lithium_demand['Initial_Lithium_Required_kg'] + total_lithium_demand['Lithium_Required_kg']
    total_lithium_demand['Cumulative_Total_Lithium_Required_kg'] = total_lithium_demand['Total_Lithium_Required_kg'].cumsum()

    return total_lithium_demand

def plot_lithium_demand(total_lithium_demand):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=total_lithium_demand, x='Year', y='Total_Lithium_Required_kg', marker='o')
    plt.title('Total Lithium Requirements Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Lithium Required (kg)')
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=total_lithium_demand, x='Year', y='Cumulative_Total_Lithium_Required_kg', marker='o')
    plt.title('Cumulative Lithium Requirements Over Time')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Lithium Required (kg)')
    plt.show()

def analyze_efficiency(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Commuter_Type', y='Avg_Daily_Energy', data=data)
    plt.title('Energy Efficiency by Commuter Type')
    plt.xlabel('Commuter Type')
    plt.ylabel('Average Daily Energy Consumption (kWh)')
    plt.show()

    efficiency = data['Total_Distance'] / data['Total_Energy_Consumed']
    plt.figure(figsize=(10, 6))
    sns.histplot(efficiency, bins=20, kde=True)
    plt.title('Distribution of Energy Efficiency')
    plt.xlabel('Efficiency (km/kWh)')
    plt.ylabel('Frequency')
    plt.show()

def main():
    plt.style.use('seaborn')
    data = load_and_preprocess_data('ev_simulation_output/results.csv')
    
    plot_commuter_types(data)
    plot_usage_duration(data)
    plot_daily_distance_vs_energy(data)
    plot_total_distance_by_commuter(data)
    plot_usage_duration_vs_energy(data)
    
    # Based on 40 kwh battery
    lithium_per_battery = 6.392  # kg
    total_lithium_demand = calculate_lithium_demand(data, lithium_per_battery)
    plot_lithium_demand(total_lithium_demand)
    
    analyze_efficiency(data)
    
    # Additional analysis: Correlation matrix
    correlation_matrix = data[['Usage_Duration', 'Total_Distance', 'Total_Energy_Consumed', 'Avg_Daily_Distance', 'Avg_Daily_Energy']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Key Variables')
    plt.show()
    
    # Prepare data for time series analysis
    monthly_replacements = data.groupby(data['Replacement_Date'].dt.to_period('M')).size().reset_index(name='Replacements')
    monthly_replacements['Replacement_Date'] = monthly_replacements['Replacement_Date'].dt.to_timestamp()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Replacement_Date', y='Replacements', data=monthly_replacements)
    plt.title('Monthly Battery Replacements Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Replacements')
    plt.show()

if __name__ == "__main__":
    main()