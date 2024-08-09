import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX



warnings.filterwarnings('ignore')

SCALING_FACTOR = 437149 / 65573  # Approximately 6.666

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
    commuter_counts = data['Commuter_Type'].value_counts() * SCALING_FACTOR
    sns.barplot(x=commuter_counts.index, y=commuter_counts.values)
    plt.title('Scaled Distribution of Commuter Types')
    plt.xlabel('Commuter Type')
    plt.ylabel('Scaled Count')
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

    # Apply scaling factor
    total_lithium_demand['Number_of_Replacements'] *= SCALING_FACTOR
    total_lithium_demand['Lithium_Required_kg'] *= SCALING_FACTOR
    total_lithium_demand['Number_of_New_EVs'] *= SCALING_FACTOR
    total_lithium_demand['Initial_Lithium_Required_kg'] *= SCALING_FACTOR
    total_lithium_demand['Total_Lithium_Required_kg'] *= SCALING_FACTOR
    total_lithium_demand['Cumulative_Total_Lithium_Required_kg'] *= SCALING_FACTOR

    return total_lithium_demand

def plot_lithium_demand(total_lithium_demand):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=total_lithium_demand, x='Year', y='Total_Lithium_Required_kg', marker='o')
    plt.title('Total Scaled Lithium Requirements Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Scaled Lithium Required (kg)')
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=total_lithium_demand, x='Year', y='Cumulative_Total_Lithium_Required_kg', marker='o')
    plt.title('Cumulative Scaled Lithium Requirements Over Time')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Scaled Lithium Required (kg)')
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

def prepare_monthly_data(data):
    monthly_replacements = data.groupby(data['Replacement_Date'].dt.to_period('M')).size().reset_index(name='Replacements')
    monthly_replacements['Replacement_Date'] = monthly_replacements['Replacement_Date'].dt.to_timestamp()
    monthly_replacements['Replacements'] *= SCALING_FACTOR  # Apply scaling factor here
    return monthly_replacements

def train_test_split(data, test_size=12):
    train = data[:-test_size]
    test = data[-test_size:]
    return train, test

def arima_forecast(train, test, order=(1, 1, 1)):
    model = ARIMA(train['Replacements'], order=order)
    results = model.fit()
    forecast = results.forecast(steps=len(test))
    return forecast

def prophet_forecast(train, test):
    prophet_data = train.rename(columns={'Replacement_Date': 'ds', 'Replacements': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_data)
    future_dates = pd.DataFrame({'ds': test['Replacement_Date']})
    forecast = model.predict(future_dates)
    return forecast['yhat']

def plot_forecasts(train, test, arima_forecast, sarima_forecast, prophet_forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train['Replacement_Date'], train['Replacements'], label='Scaled Training Data')
    plt.plot(test['Replacement_Date'], test['Replacements'], label='Scaled Actual Data')
    plt.plot(test['Replacement_Date'], arima_forecast, label='Scaled ARIMA Forecast')
    plt.plot(test['Replacement_Date'], sarima_forecast, label='Scaled SARIMA Forecast')
    plt.plot(test['Replacement_Date'], prophet_forecast, label='Scaled Prophet Forecast')
    plt.title('Scaled Battery Replacement Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Number of Replacements (Scaled)')
    plt.legend()
    plt.show()

def long_term_arima_forecast(data, periods=600):  # 50 years * 12 months
    model = ARIMA(data['Replacements'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=periods)
    return forecast  # Already scaled as input data is scaled

def long_term_prophet_forecast(data, periods=600):
    prophet_data = data.rename(columns={'Replacement_Date': 'ds', 'Replacements': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.add_country_holidays(country_name='CA')
    model.fit(prophet_data)
    future_dates = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future_dates)
    # Return only the future periods
    return forecast.iloc[-periods:]

def plot_long_term_forecasts(data, long_term_dates, arima_forecast, sarima_forecast, prophet_forecast):
    plt.figure(figsize=(20, 10))
    plt.plot(data['Replacement_Date'], data['Replacements'], label='Scaled Historical Data')
    plt.plot(long_term_dates, arima_forecast, label='Scaled ARIMA Forecast')
    plt.plot(long_term_dates, sarima_forecast, label='Scaled SARIMA Forecast')
    plt.plot(long_term_dates, prophet_forecast['yhat'], label='Scaled Prophet Forecast')
    plt.title('50-Year Scaled Battery Replacement Forecasts')
    plt.xlabel('Date')
    plt.ylabel('Number of Replacements (Scaled)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def sarima_forecast(train, test, order=(0,1,0), seasonal_order=(1,1,2,12)):
    model = SARIMAX(train['Replacements'], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    forecast = results.forecast(steps=len(test))
    return forecast

def long_term_sarima_forecast(data, periods=600):  # 50 years * 12 months
    model = SARIMAX(data['Replacements'], order=(0,1,0), seasonal_order=(1,1,2,12))
    results = model.fit()
    forecast = results.forecast(steps=periods)
    return forecast

def evaluate_model(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8)) * 100)
    return mae, rmse, mape

def main():
    plt.style.use('seaborn')
    data = load_and_preprocess_data('ev_simulation_output/results.csv')
    
    plot_commuter_types(data)
    #plot_usage_duration(data)
    #plot_daily_distance_vs_energy(data)
    plot_total_distance_by_commuter(data)
    plot_usage_duration_vs_energy(data)
    
    lithium_per_battery = 6.392  # kg (based on 40 kWh battery)
    total_lithium_demand = calculate_lithium_demand(data, lithium_per_battery)
    plot_lithium_demand(total_lithium_demand)
    
    #analyze_efficiency(data)
    
    # correlation_matrix = data[['Usage_Duration', 'Total_Distance', 'Total_Energy_Consumed', 'Avg_Daily_Distance', 'Avg_Daily_Energy']].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Correlation Matrix of Key Variables')
    # plt.show()
    
    monthly_data = prepare_monthly_data(data)
    train, test = train_test_split(monthly_data)
    
    arima_forecast_values = arima_forecast(train, test)
    sarima_forecast_values = sarima_forecast(train, test)
    prophet_forecast_values = prophet_forecast(train, test)
    
    plot_forecasts(train, test, arima_forecast_values, sarima_forecast_values, prophet_forecast_values)
    
    arima_mae, arima_rmse, arima_mape = evaluate_model(test['Replacements'].values, arima_forecast_values)
    sarima_mae, sarima_rmse, sarima_mape = evaluate_model(test['Replacements'].values, sarima_forecast_values)
    prophet_mae, prophet_rmse, prophet_mape = evaluate_model(test['Replacements'].values, prophet_forecast_values)

    print("ARIMA Model Evaluation:")
    print(f"MAE: {arima_mae:.2f}")
    print(f"RMSE: {arima_rmse:.2f}")
    print(f"MAPE: {arima_mape:.2f}%")

    print("\nSARIMA Model Evaluation:")
    print(f"MAE: {sarima_mae:.2f}")
    print(f"RMSE: {sarima_rmse:.2f}")
    print(f"MAPE: {sarima_mape:.2f}%")
    
    print("\nProphet Model Evaluation:")
    print(f"MAE: {prophet_mae:.2f}")
    print(f"RMSE: {prophet_rmse:.2f}")
    print(f"MAPE: {prophet_mape:.2f}%")
    
    # Short-term forecasts (2 years)
    future_periods = 24
    last_date = monthly_data['Replacement_Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='M')
    
    arima_future_forecast = long_term_arima_forecast(monthly_data, periods=future_periods)
    sarima_future_forecast = long_term_sarima_forecast(monthly_data, periods=future_periods)
    prophet_future_forecast = long_term_prophet_forecast(monthly_data, periods=future_periods)
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data['Replacement_Date'], monthly_data['Replacements'], label='Scaled Historical Data')
    plt.plot(future_dates, arima_future_forecast, label='Scaled ARIMA Future Forecast')
    plt.plot(future_dates, sarima_future_forecast, label='Scaled SARIMA Future Forecast')
    plt.plot(future_dates, prophet_future_forecast['yhat'], label='Scaled Prophet Future Forecast')
    plt.title('Scaled Future Battery Replacement Forecasts (2 Years)')
    plt.xlabel('Date')
    plt.ylabel('Number of Replacements (Scaled)')
    plt.legend()
    plt.show()

    print("\nScaled Future Forecasts (2 Years):")
    future_forecasts = pd.DataFrame({
        'Date': future_dates,
        'ARIMA_Forecast': arima_future_forecast,
        'SARIMA_Forecast': sarima_future_forecast,
        'Prophet_Forecast': prophet_future_forecast['yhat'].values
    })
    print(future_forecasts.to_string(index=False))

    # Long-term forecasting (50 years)
    print("Generating 50-year scaled forecasts...")
    last_date = monthly_data['Replacement_Date'].max()
    long_term_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=600, freq='M')
    
    arima_long_forecast = long_term_arima_forecast(monthly_data)
    sarima_long_forecast = long_term_sarima_forecast(monthly_data)
    prophet_long_forecast = long_term_prophet_forecast(monthly_data)
    
    plot_long_term_forecasts(monthly_data, long_term_dates, arima_long_forecast, sarima_long_forecast, prophet_long_forecast)

    # Calculate and plot cumulative lithium demand based on forecasts
    lithium_per_battery = 6.392  # kg
    arima_cumulative_demand = pd.DataFrame({
        'Date': long_term_dates,
        'Cumulative_Lithium_Demand_kg': (arima_long_forecast.cumsum() * lithium_per_battery).values
    })
    
    sarima_cumulative_demand = pd.DataFrame({
        'Date': long_term_dates,
        'Cumulative_Lithium_Demand_kg': (sarima_long_forecast.cumsum() * lithium_per_battery).values
    })
    
    prophet_cumulative_demand = prophet_long_forecast[['ds', 'yhat']].copy()
    prophet_cumulative_demand['Cumulative_Lithium_Demand_kg'] = prophet_cumulative_demand['yhat'].cumsum() * lithium_per_battery

    plt.figure(figsize=(20, 10))
    plt.plot(arima_cumulative_demand['Date'], arima_cumulative_demand['Cumulative_Lithium_Demand_kg'], 
             label='Scaled ARIMA Forecast')
    plt.plot(sarima_cumulative_demand['Date'], sarima_cumulative_demand['Cumulative_Lithium_Demand_kg'], 
             label='Scaled SARIMA Forecast')
    plt.plot(prophet_cumulative_demand['ds'], prophet_cumulative_demand['Cumulative_Lithium_Demand_kg'], 
             label='Scaled Prophet Forecast')
    plt.title('50-Year Scaled Cumulative Lithium Demand Forecast')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Lithium Demand (kg)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary of 50-Year Scaled Forecasts:")
    print(f"ARIMA Total Predicted Replacements: {arima_long_forecast.sum():,.0f}")
    print(f"SARIMA Total Predicted Replacements: {sarima_long_forecast.sum():,.0f}")
    print(f"Prophet Total Predicted Replacements: {prophet_long_forecast['yhat'].sum():,.0f}")
    print(f"ARIMA Cumulative Lithium Demand: {arima_cumulative_demand['Cumulative_Lithium_Demand_kg'].iloc[-1]:,.0f} kg")
    print(f"SARIMA Cumulative Lithium Demand: {sarima_cumulative_demand['Cumulative_Lithium_Demand_kg'].iloc[-1]:,.0f} kg")
    print(f"Prophet Cumulative Lithium Demand: {prophet_cumulative_demand['Cumulative_Lithium_Demand_kg'].iloc[-1]:,.0f} kg")



if __name__ == "__main__":
    main()