import argparse
import numpy as np
import csv
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
import os
import shutil

class ElectricVehicle:
    def __init__(self, battery_size, max_soc, min_soc, consumption, commuter_type, battery_replacement_threshold, wfh_days):
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

    def compute_SOC_arr(self, dist_km):
        used_kwh = (dist_km * self.consumption) / 1000  
        soc_change = used_kwh / self.battery_size
        return max(self.max_soc * self.battery_size - soc_change, self.min_soc * self.battery_size)
    
    def degrade(self):
        self.battery_size *= (1 - self.degradation_rate)
        if self.battery_size / self.initial_battery_size <= self.battery_replacement_threshold:
            self.replace_battery()

    def replace_battery(self):
        self.battery_size = self.initial_battery_size
        self.battery_replacements += 1

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
        ev.degrade()
        day += 1

    return trip_data

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

def simulate_vehicle(commuter_type, battery_replacement_threshold, vehicle_id, output_folder):
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

    ev = ElectricVehicle(ev_battery, max_soc, min_soc, consumption, commuter_type, battery_replacement_threshold, wfh_days)
    trip_data = generate_trip_data(ev, C_dist, C_dept, C_arr, N_nc)
    
    filename = f"{commuter_type}_{int(battery_replacement_threshold*100)}_{vehicle_id}.csv"
    filepath = os.path.join(output_folder, filename)
    
    write_to_csv(filepath, trip_data)
    
    return ev.battery_replacements

def run_simulation(output_folder, num_vehicles):
    clean_output_folder(output_folder)

    commuter_types = ["Classic", "Hybrid", "Freelancer"]
    commuter_probabilities = [0.756, 0.103, 0.141]  # Exact probabilities as per the requirements for 2023
    battery_thresholds = [0.8, 0.7, 0.6]
    battery_threshold_probabilities = [0.1, 0.2, 0.7]

    simulation_params = []
    battery_threshold_counts = Counter()

    for i in range(num_vehicles):
        commuter_type = np.random.choice(commuter_types, p=commuter_probabilities)
        battery_threshold = np.random.choice(battery_thresholds, p=battery_threshold_probabilities)
        simulation_params.append((commuter_type, battery_threshold, i, output_folder))
        battery_threshold_counts[battery_threshold] += 1

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(simulate_vehicle, *zip(*simulation_params)))
    
    total_replacements = sum(results)
    print(f"Total battery replacements: {total_replacements}")

    # Count the distribution of commuter types
    commuter_counts = {t: sum(1 for p in simulation_params if p[0] == t) for t in commuter_types}
    print("\nActual Commuter Type Distribution:")
    for t, count in commuter_counts.items():
        print(f"{t}: {count} ({count/num_vehicles*100:.1f}%)")

    # Print battery replacement threshold distribution
    print("\nBattery Replacement Threshold Distribution:")
    for threshold, count in battery_threshold_counts.items():
        print(f"{threshold*100}%: {count} ({count/num_vehicles*100:.1f}%)")

    # Validate battery replacement thresholds
    expected_threshold_counts = {0.8: num_vehicles * 0.1, 0.7: num_vehicles * 0.2, 0.6: num_vehicles * 0.7}
    print("\nBattery Replacement Threshold Validation:")
    for threshold, expected_count in expected_threshold_counts.items():
        actual_count = battery_threshold_counts[threshold]
        difference = actual_count - expected_count
        print(f"{threshold*100}%: Expected {expected_count:.0f}, Actual {actual_count}, Difference {difference:.0f}")

def main():
    parser = argparse.ArgumentParser(description='Sample synthetic EV usage data for multiple vehicles.')
    parser.add_argument('--output_folder', type=str, default='ev_simulation_output', help='Output folder name')
    parser.add_argument('--num_vehicles', type=int, default=100, help='Number of vehicles to simulate')

    args = parser.parse_args()
    run_simulation(args.output_folder, args.num_vehicles)

if __name__ == '__main__':
    main()
