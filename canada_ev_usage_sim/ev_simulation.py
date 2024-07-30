import argparse
from datetime import datetime, timedelta
import numpy as np
import csv
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
import os
import shutil
from statistics import StatisticsError, mean
import time

class ElectricVehicle:
    def __init__(self, battery_size, max_soc, min_soc, consumption, commuter_type, battery_replacement_threshold, wfh_days, start_date):
        self.battery_size = battery_size
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.consumption = consumption
        self.degradation_rate = 0.00006301369
        self.commuter_type = commuter_type
        self.battery_replacement_threshold = battery_replacement_threshold
        self.initial_battery_size = battery_size
        self.battery_replacements = 0
        self.wfh_days = wfh_days
        self.battery_start_day = 0
        self.battery_lifespans = []
        self.current_battery_age = 0
        self.start_date = start_date
        self.replacement_date = None
        self.capacity_at_replacement = None        

    def update_usage(self, distance, energy):
        self.total_distance += distance
        self.total_energy_consumed += energy

    def compute_SOC_arr(self, dist_km):
        used_kwh = (dist_km * self.consumption) / 1000  
        soc_change = used_kwh / self.battery_size
        return max(self.max_soc * self.battery_size - soc_change, self.min_soc * self.battery_size)
    
    def degrade(self, current_day):
        self.battery_size *= (1 - self.degradation_rate)
        self.current_battery_age = current_day - self.battery_start_day
        if self.battery_size / self.initial_battery_size <= self.battery_replacement_threshold:
            self.replace_battery(current_day)

    def replace_battery(self, current_day):
        self.capacity_at_replacement = self.battery_size
        self.battery_size = self.initial_battery_size
        self.battery_replacements += 1
        self.replacement_date = self.start_date + timedelta(days=current_day)
        self.battery_start_day = current_day
        self.current_battery_age = 0


# Sample data for quarterly registrations based on total number scaled down to .15
# QUARTERLY_DATA = [
#     ("2017-01-01", 250),
#     ("2017-04-01", 329),
#     ("2017-07-01", 360),
#     ("2017-10-01", 423),
#     ("2018-01-01", 396),
#     ("2018-04-01", 1109),
#     ("2018-07-01", 976),
#     ("2018-10-01", 905),
#     ("2019-01-01", 791),
#     ("2019-04-01", 1856),
#     ("2019-07-01", 1518),
#     ("2019-10-01", 1164),
#     ("2020-01-01", 1258),
#     ("2020-04-01", 913),
#     ("2020-07-01", 1890),
#     ("2020-10-01", 1794),
#     ("2021-01-01", 1904),
#     ("2021-04-01", 2425),
#     ("2021-07-01", 2377),
#     ("2021-10-01", 2103),
#     ("2022-01-01", 2954),
#     ("2022-04-01", 3265),
#     ("2022-07-01", 4406),
#     ("2022-10-01", 4163),
#     ("2023-01-01", 3569),
#     ("2023-04-01", 5192),
#     ("2023-07-01", 6388),
#     ("2023-10-01", 5779),
#     ("2024-01-01", 5116)
# ]

# Small sample data for testing
QUARTERLY_DATA = [
    ("2017-01-01", 100),
    ("2017-04-01", 100),
    ("2017-07-01", 100),
    ("2017-10-01", 100),
    ("2018-01-01", 100),
    ("2018-04-01", 100),
    ("2018-07-01", 100),
    ("2018-10-01", 100),
    ("2019-01-01", 100),
    ("2019-04-01", 100),
    ("2019-07-01", 100),
    ("2019-10-01", 100),
    ("2020-01-01", 100),
    ("2020-04-01", 100),
    ("2020-07-01", 100),
    ("2020-10-01", 100),
    ("2021-01-01", 100),
    ("2021-04-01", 100),
    ("2021-07-01", 100),
    ("2021-10-01", 100),
    ("2022-01-01", 100),
    ("2022-04-01", 100),
    ("2022-07-01", 100),
    ("2022-10-01", 100),
    ("2023-01-01", 100),
    ("2023-04-01", 100),
    ("2023-07-01", 100),
    ("2023-10-01", 100),
    ("2024-01-01", 100)
]

def clean_output_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def generate_trip_data(ev, C_dist, C_dept, C_arr, N_nc):
    trip_data = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def reduce_soc(distance_km, current_soc):
        energy_used = (distance_km * ev.consumption) / 1000
        soc_after_trip = current_soc - energy_used
        return max(ev.min_soc * ev.battery_size, soc_after_trip)

    average_non_commute_trips_per_day = (N_nc / 2) / 7
    average_speed_kmh = 50
    total_energy_consumed = 0
    total_distance = 0
    total_throughput = ev.battery_size * 1000
    day = 0

    while (total_distance < 321000 and total_energy_consumed < total_throughput and ev.battery_replacements < 1):
        week_day = day % 7
        is_wfh_day = ev.wfh_days[week_day]
        is_weekend = week_day in [5, 6]
        current_soc = ev.max_soc * ev.battery_size
        trips_today = []

        if not is_wfh_day and not is_weekend:
            commute_dist = random.uniform(C_dist - C_dist * 0.1, C_dist + C_dist * 0.1)
            t_dep, t_arr = sample_commute_times(C_dept, C_arr)
            soc_start = current_soc
            soc_end = reduce_soc(commute_dist, soc_start)
            commute_travel_time = (C_arr - C_dept) * 60
            energy_consumed = soc_start - soc_end
            total_energy_consumed += energy_consumed
            total_distance += commute_dist
            trips_today.append((weekdays[week_day], format_time(t_dep), f"{soc_start:.2f}", format_time(t_arr), f"{soc_end:.2f}", f"{commute_dist:.2f}", round(commute_travel_time), f"{energy_consumed:.2f}", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}", f"{ev.battery_size:.10f}", ev.battery_replacements))
            current_soc = soc_end

        num_non_commute_trips = np.random.poisson(average_non_commute_trips_per_day)
        for _ in range(num_non_commute_trips):
            t_dep, t_arr = sample_non_commute_times()
            travel_time_hours = (t_arr - t_dep) / 5
            non_commute_dist = travel_time_hours * average_speed_kmh / 2
            soc_start = current_soc
            soc_end = reduce_soc(non_commute_dist, soc_start)
            non_commute_travel_time = travel_time_hours * 60
            energy_consumed = soc_start - soc_end
            total_energy_consumed += energy_consumed
            total_distance += non_commute_dist
            trips_today.append((weekdays[week_day], format_time(t_dep), f"{soc_start:.2f}", format_time(t_arr), f"{soc_end:.2f}", f"{non_commute_dist:.2f}", round(non_commute_travel_time), f"{energy_consumed:.2f}", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}", f"{ev.battery_size:.10f}", ev.battery_replacements))
            current_soc = soc_end

        if not trips_today:
            trips_today.append((weekdays[week_day], "No trips", f"{ev.max_soc * ev.battery_size:.2f}", "", f"{ev.max_soc * ev.battery_size:.2f}", "", "", "0.00", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}", f"{ev.battery_size:.10f}", ev.battery_replacements))

        trip_data.append((day + 1, trips_today))
        ev.degrade(day)
        day += 1

    # Add the final battery lifespan
    ev.battery_lifespans.append(day)

    return trip_data, day, total_distance, total_energy_consumed

def format_time(time_float):
    hours = int(time_float)
    minutes = int((time_float - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def sample_commute_times(C_dept, C_arr):
    t_dep_hour = random.uniform(C_dept - 0.25, C_dept + 0.25)
    t_arr_hour = random.uniform(C_arr - 0.25, C_arr + 0.25)
    
    while t_arr_hour <= t_dep_hour:
        t_arr_hour = random.uniform(C_arr - 0.25, C_arr + 0.25)

    return t_dep_hour, t_arr_hour

def sample_non_commute_times():
    t_dep_hour = random.uniform(8, 20)
    trip_duration = random.uniform(0.5, 2)
    t_arr_hour = t_dep_hour + trip_duration

    if t_arr_hour >= 24:
        t_arr_hour -= 24

    return t_dep_hour, t_arr_hour

def write_to_csv(file_name, data):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Day", "Weekday", "Departure Time", "SOC on Departure", "Arrival Time", "SOC on Arrival", "Distance (km)", "Travel Time (min)", "Energy Consumed (kWh)", "Total Energy Consumed (kWh)", "Total Distance (km)", "Battery Size", "Battery Replacements"])
        for day, trips in data:
            for trip in trips:
                writer.writerow([day, *trip])

def simulate_vehicle(commuter_type, battery_replacement_threshold, vehicle_id, output_folder, start_date):
    ev_battery = 40
    max_soc = 0.8
    min_soc = 0.2
    consumption = 164
    C_dist = 17.4
    C_dept = 8.00
    C_arr = 18.00
    N_nc = 9

    if commuter_type == "Classic":
        wfh_days = [0, 0, 0, 0, 0, 0, 0]
    elif commuter_type == "Hybrid":
        wfh_days = [1, 0, 1, 0, 1, 0, 0]
    else:  # Freelancer
        wfh_days = [1, 1, 1, 1, 1, 0, 0]

    ev = ElectricVehicle(ev_battery, max_soc, min_soc, consumption, commuter_type, battery_replacement_threshold, wfh_days, start_date)
    trip_data, total_days, total_distance, total_energy_consumed = generate_trip_data(ev, C_dist, C_dept, C_arr, N_nc)
    
    filename = f"{commuter_type}_{int(battery_replacement_threshold*100)}_{vehicle_id}.csv"
    filepath = os.path.join(output_folder, filename)
    
    write_to_csv(filepath, trip_data)
    
    end_date = start_date + timedelta(days=total_days)
    return (ev.battery_replacements, ev.battery_lifespans, total_days, ev.replacement_date, 
            filename, end_date, start_date, commuter_type, battery_replacement_threshold, 
            ev.capacity_at_replacement, total_distance, total_energy_consumed)

def write_results_csv(results, output_folder):
    results_file = os.path.join(output_folder, "results.csv")
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["EV_Name", "Start_Date", "End_Date", "Replacement_Date", "Commuter_Type", 
                         "Battery_Threshold", "Capacity_at_Replacement", "Total_Distance", "Total_Energy_Consumed"])
        for result in results:
            ev_name = result[4].split('.')[0]  # Remove .csv extension
            start_date = result[6].strftime('%Y-%m-%d')
            end_date = result[5].strftime('%Y-%m-%d')
            replacement_date = result[3].strftime('%Y-%m-%d') if result[3] else "No Replacement"
            commuter_type = result[7]
            battery_threshold = result[8]
            capacity_at_replacement = f"{result[9]:.2f}" if result[9] is not None else "N/A"
            total_distance = f"{result[10]:.2f}"
            total_energy_consumed = f"{result[11]:.2f}"
            writer.writerow([ev_name, start_date, end_date, replacement_date, commuter_type, 
                             battery_threshold, capacity_at_replacement, total_distance, total_energy_consumed])

def run_simulation(output_folder):
    clean_output_folder(output_folder)

    commuter_types = ["Classic", "Hybrid", "Freelancer"]
    commuter_probabilities = [0.817, 0.041, 0.142]  # Average of probabilities from 2017 to 2023
    battery_thresholds = [0.8, 0.7, 0.6]
    battery_threshold_probabilities = [0.1, 0.2, 0.7]

    all_results = []
    vehicle_id = 0
    battery_lifespans = {0.8: [], 0.7: [], 0.6: []}
    battery_threshold_counts = Counter()
    commuter_counts = Counter()

    for quarter_start, num_evs in QUARTERLY_DATA:
        quarter_start_date = datetime.strptime(quarter_start, "%Y-%m-%d")
        
        for _ in range(num_evs):
            # Generate a random start date within the quarter
            random_days = random.randint(0, 89)  # 0 to 89 days (90 days per quarter)
            start_date = quarter_start_date + timedelta(days=random_days)

            commuter_type = np.random.choice(commuter_types, p=commuter_probabilities)
            battery_threshold = np.random.choice(battery_thresholds, p=battery_threshold_probabilities)
            
            result = simulate_vehicle(commuter_type, battery_threshold, vehicle_id, output_folder, start_date)
            all_results.append(result)
            
            battery_lifespans[battery_threshold].extend(result[1])
            battery_threshold_counts[battery_threshold] += 1
            commuter_counts[commuter_type] += 1
            
            vehicle_id += 1

    write_results_csv(all_results, output_folder)

    total_replacements = sum(replacements for replacements, _, _, _, _, _, _, _, _, _, _, _ in all_results)
    all_lifespans = [lifespan for _, lifespans, _, _, _, _, _, _, _, _, _, _ in all_results for lifespan in lifespans]
    total_days = [days for _, _, days, _, _, _, _, _, _, _, _, _ in all_results]
    total_vehicles = sum(num_evs for _, num_evs in QUARTERLY_DATA)

    print(f"Total battery replacements: {total_replacements}")
    
    if all_lifespans:
        print(f"\nMedian battery lifespan: {np.median(all_lifespans) / 365:.2f} years")
        try:
            print(f"Mean battery lifespan: {mean(all_lifespans) / 365:.2f} years")
        except StatisticsError:
            print("Mean battery lifespan: Unable to calculate (no data)")
    else:
        print("\nNo battery lifespan data available")


    # Print median and mean lifespan by replacement threshold
    print("\nLifespan by replacement threshold:")
    for threshold, lifespans in battery_lifespans.items():
        if lifespans:
            med = np.median(lifespans) / 365  # Convert days to years
            avg = mean(lifespans) / 365  # Convert days to years
            print(f"{threshold*100}%: Median {med:.2f} years, Mean {avg:.2f} years")
        else:
            print(f"{threshold*100}%: No data")

    # Print average simulation duration
    avg_simulation_duration = mean(total_days) / 365  # Convert days to years
    print(f"\nAverage simulation duration: {avg_simulation_duration:.2f} years")

    # Count the distribution of commuter types
    print("\nActual Commuter Type Distribution:")
    for t, count in commuter_counts.items():
        print(f"{t}: {count} ({count/total_vehicles*100:.1f}%)")

    # Print battery replacement threshold distribution
    print("\nBattery Replacement Threshold Distribution:")
    for threshold, count in battery_threshold_counts.items():
        print(f"{threshold*100}%: {count} ({count/total_vehicles*100:.1f}%)")

    # Validate battery replacement thresholds
    expected_threshold_counts = {0.8: total_vehicles * 0.1, 0.7: total_vehicles * 0.2, 0.6: total_vehicles * 0.7}
    print("\nBattery Replacement Threshold Validation:")
    for threshold, expected_count in expected_threshold_counts.items():
        actual_count = battery_threshold_counts[threshold]
        difference = actual_count - expected_count
        print(f"{threshold*100}%: Expected {expected_count:.0f}, Actual {actual_count}, Difference {difference:.0f}")


def main():
    parser = argparse.ArgumentParser(description='Sample synthetic EV usage data for multiple vehicles based on quarterly registrations.')
    parser.add_argument('--output_folder', type=str, default='ev_simulation_output', help='Output folder name')

    args = parser.parse_args()

    run_simulation(args.output_folder)

if __name__ == '__main__':
    main()