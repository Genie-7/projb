# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from prophet import Prophet
# import warnings

# warnings.filterwarnings('ignore')

# SCALING_FACTOR = 437149 / 65573  # Approximately 6.666
# LITHIUM_PER_BATTERY = 6.392  # kg (based on 40 kWh battery)

# def load_and_preprocess_data(file_path):
#     data = pd.read_csv(file_path)
#     date_columns = ['Start_Date', 'End_Date', 'Replacement_Date']
#     for col in date_columns:
#         data[col] = pd.to_datetime(data[col])
#     return data

# def calculate_cumulative_lithium_demand(data):
#     start_events = data[['Start_Date']].copy()
#     start_events['Lithium_Required_kg'] = LITHIUM_PER_BATTERY

#     replacement_events = data[['Replacement_Date']].copy()
#     replacement_events['Lithium_Required_kg'] = LITHIUM_PER_BATTERY

#     demand_events = pd.concat([start_events.rename(columns={'Start_Date': 'Date'}), 
#                                replacement_events.rename(columns={'Replacement_Date': 'Date'})])
    
#     demand_events['Lithium_Required_kg'] *= SCALING_FACTOR
#     demand_events = demand_events.sort_values(by='Date')
#     demand_events['Cumulative_Lithium_Required_kg'] = demand_events['Lithium_Required_kg'].cumsum()
    
#     daily_demand = demand_events.resample('D', on='Date').last().reset_index()
#     daily_demand['Cumulative_Lithium_Required_kg'].fillna(method='ffill', inplace=True)
    
#     return daily_demand

# def forecast_cumulative_demand(data, periods):
#     model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
#     model.fit(data)
#     future = model.make_future_dataframe(periods=periods, freq='D')
#     forecast = model.predict(future)
#     return forecast[['ds', 'yhat', 'yhat_upper']]

# def plot_cumulative_demand(data, forecast, period):
#     plt.figure(figsize=(14, 8))
#     plt.plot(data['ds'], data['y'], label='Historical Data')
#     plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
#     plt.plot(forecast['ds'], forecast['yhat_upper'], label='Upper Bound', linestyle=':', color='red')
#     plt.fill_between(forecast['ds'], forecast['yhat'], forecast['yhat_upper'], alpha=0.2)
#     plt.title(f'Cumulative Lithium Demand Forecast ({period} years)')
#     plt.xlabel('Year')
#     plt.ylabel('Cumulative Lithium Demand (kg)')
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.show()

# def main():
#     data = load_and_preprocess_data('ev_simulation_output/results.csv')
#     daily_demand = calculate_cumulative_lithium_demand(data)
#     daily_demand = daily_demand.rename(columns={'Date': 'ds', 'Cumulative_Lithium_Required_kg': 'y'})
    
#     # Plot historical cumulative demand
#     plt.figure(figsize=(14, 8))
#     plt.plot(daily_demand['ds'], daily_demand['y'])
#     plt.title('Historical Cumulative Lithium Demand')
#     plt.xlabel('Date')
#     plt.ylabel('Cumulative Lithium Demand (kg)')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.show()
    
#     periods = [2*365, 5*365, 10*365, 25*365, 50*365]  # Convert years to days
#     for period in periods:
#         forecast = forecast_cumulative_demand(daily_demand, period)
#         plot_cumulative_demand(daily_demand, forecast, period // 365)
        
#         # Output upper bound prediction at the end of the period
#         end_date = forecast['ds'].max()
#         upper_bound_prediction = forecast[forecast['ds'] == end_date]['yhat_upper'].values[0]
#         print(f"Upper bound prediction for {period // 365} years in the future ({end_date.date()}): {upper_bound_prediction:.2f} kg")

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools
import warnings
import json
import os

warnings.filterwarnings('ignore')

SCALING_FACTOR = 437149 / 65573  # Approximately 6.666
LITHIUM_PER_BATTERY = 6.392  # kg (based on 40 kWh battery)
PARAMS_FILE = 'best_params.json'

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    date_columns = ['Start_Date', 'End_Date', 'Replacement_Date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])
    return data

def calculate_cumulative_lithium_demand(data):
    start_events = data[['Start_Date']].copy()
    start_events['Lithium_Required_kg'] = LITHIUM_PER_BATTERY

    replacement_events = data[['Replacement_Date']].copy()
    replacement_events['Lithium_Required_kg'] = LITHIUM_PER_BATTERY

    demand_events = pd.concat([start_events.rename(columns={'Start_Date': 'Date'}), 
                               replacement_events.rename(columns={'Replacement_Date': 'Date'})])
    
    demand_events['Lithium_Required_kg'] *= SCALING_FACTOR
    demand_events = demand_events.sort_values(by='Date')
    demand_events['Cumulative_Lithium_Required_kg'] = demand_events['Lithium_Required_kg'].cumsum()
    
    daily_demand = demand_events.resample('D', on='Date').last().reset_index()
    daily_demand['Cumulative_Lithium_Required_kg'].fillna(method='ffill', inplace=True)
    
    return daily_demand

def optimize_hyperparameters(data):
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []

    total_combinations = len(all_params)
    for i, params in enumerate(all_params, 1):
        print(f"Testing parameter combination {i}/{total_combinations}")
        m = Prophet(**params).fit(data)
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print(tuning_results)
    best_params = all_params[np.argmin(rmses)]
    
    print(f'Best parameters: {best_params}')
    return best_params

def save_params(params):
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params, f)

def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f)
    return None

def forecast_cumulative_demand(data, periods, best_params):
    model = Prophet(**best_params)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_upper']]

def plot_cumulative_demand(data, forecast, period):
    plt.figure(figsize=(14, 8))
    plt.plot(data['ds'], data['y'], label='Historical Data')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.plot(forecast['ds'], forecast['yhat_upper'], label='Upper Bound', linestyle=':', color='red')
    plt.fill_between(forecast['ds'], forecast['yhat'], forecast['yhat_upper'], alpha=0.2)
    plt.title(f'Cumulative Lithium Demand Forecast ({period} years)')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Lithium Demand (kg)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main():
    data = load_and_preprocess_data('ev_simulation_output/results.csv')
    daily_demand = calculate_cumulative_lithium_demand(data)
    daily_demand = daily_demand.rename(columns={'Date': 'ds', 'Cumulative_Lithium_Required_kg': 'y'})
    
    plt.figure(figsize=(14, 8))
    plt.plot(daily_demand['ds'], daily_demand['y'])
    plt.title('Historical Cumulative Lithium Demand')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Lithium Demand (kg)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    best_params = load_params()
    if best_params is None:
        optimize = input("No saved parameters found. Do you want to optimize hyperparameters? (y/n): ").lower()
        if optimize == 'y':
            best_params = optimize_hyperparameters(daily_demand)
            save_params(best_params)
        else:
            print("Using default parameters.")
            best_params = {}
    else:
        print("Loaded parameters from file.")
    
    periods = [2*365, 5*365, 10*365, 25*365, 50*365]  # Convert years to days
    for period in periods:
        forecast = forecast_cumulative_demand(daily_demand, period, best_params)
        plot_cumulative_demand(daily_demand, forecast, period // 365)
        
        end_date = forecast['ds'].max()
        upper_bound_prediction = forecast[forecast['ds'] == end_date]['yhat_upper'].values[0]
        print(f"Upper bound prediction for {period // 365} years in the future ({end_date.date()}): {upper_bound_prediction:.2f} kg")

if __name__ == "__main__":
    main()