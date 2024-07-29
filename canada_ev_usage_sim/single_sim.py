import argparse
import numpy as np
import csv
import random

class ElectricVehicle:
    def __init__(self, battery_size, max_soc, min_soc, consumption):
        self.battery_size = battery_size
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.consumption = consumption
        self.degradation_rate = 0.00006301369

    def compute_SOC_arr(self, dist_km):
        used_kwh = (dist_km * self.consumption) / 1000  
        soc_change = used_kwh / self.battery_size
        return max(self.max_soc * self.battery_size - soc_change, self.min_soc * self.battery_size)
    
    def degrade(self):
        self.battery_size *= (1 - self.degradation_rate)

def validate_input(args):
    if not (0 <= args.max_soc <= 1 and 0 <= args.min_soc <= 1):
        raise ValueError("SOC values must be between 0 and 1")
    if args.max_soc < args.min_soc:
        raise ValueError("Max SOC must be greater than Min SOC")
    if args.ev_battery <= 0:
        raise ValueError("EV battery size must be a positive number")
    if args.consumption <= 0:
        raise ValueError("EV consumption must be a positive number")
    if not (0 <= args.C_dept < 24 and 0 <= args.C_arr < 24):
        raise ValueError("Departure and arrival times must be within 24-hour range")
    if args.C_dept >= args.C_arr:
        raise ValueError("Departure time must be before arrival time")
    for day in [args.wfh_monday, args.wfh_tuesday, args.wfh_wednesday, args.wfh_thursday, args.wfh_friday]:
        if day not in [0, 1]:
            raise ValueError("WFH day inputs must be either 0 or 1")
    if args.C_dist <= 0 :
        raise ValueError("Commute distance must be positive")
    if args.N_nc < 0:
        raise ValueError("Number of weekly non-commuting trips must be non-negative")


def generate_trip_data(args, ev):
    trip_data = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wfh_days = {
        0: args.wfh_monday,
        1: args.wfh_tuesday,
        2: args.wfh_wednesday,
        3: args.wfh_thursday,
        4: args.wfh_friday
    }

    def reduce_soc(distance_km, current_soc):
        energy_used = (distance_km * ev.consumption) / 1000  # Convert Wh to kWh
        soc_after_trip = current_soc - energy_used
        return max(ev.min_soc * ev.battery_size, soc_after_trip)

    average_non_commute_trips_per_day = (args.N_nc / 2) / 7
    average_speed_kmh = 50
    total_energy_consumed = 0
    total_distance = 0
    total_throughput = ev.battery_size * 1000
    deg_breakpoint = ev.battery_size * 0.6
    print(deg_breakpoint)
    day = 0

    while (total_distance < 321000 and total_energy_consumed < total_throughput and ev.battery_size > deg_breakpoint):
        week_day = day % 7
        is_wfh_day = wfh_days.get(week_day, 0) == 1
        is_weekend = week_day in [5, 6]
        current_soc = ev.max_soc * ev.battery_size
        trips_today = []

        if not is_wfh_day and not is_weekend:
            commute_dist = random.uniform(args.C_dist - args.C_dist * 0.1, args.C_dist + args.C_dist * 0.1)
            t_dep, t_arr = sample_commute_times(args)
            soc_start = current_soc
            soc_end = reduce_soc(commute_dist, soc_start)
            commute_travel_time = (args.C_arr - args.C_dept) * 60
            energy_consumed = soc_start - soc_end  # Corrected calculation
            total_energy_consumed += energy_consumed
            total_distance += commute_dist
            trips_today.append((weekdays[week_day], format_time(t_dep), f"{soc_start:.2f}", format_time(t_arr), f"{soc_end:.2f}", f"{commute_dist:.2f}", round(commute_travel_time), f"{energy_consumed:.2f}", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}", f"{ev.battery_size:.10f}"))
            current_soc = soc_end

        num_non_commute_trips = np.random.poisson(average_non_commute_trips_per_day)
        for _ in range(num_non_commute_trips):
            t_dep, t_arr = sample_non_commute_times(args)
            travel_time_hours = (t_arr - t_dep) / 5
            non_commute_dist = travel_time_hours * average_speed_kmh / 2
            soc_start = current_soc
            soc_end = reduce_soc(non_commute_dist, soc_start)
            non_commute_travel_time = travel_time_hours * 60
            energy_consumed = soc_start - soc_end  # Corrected calculation
            total_energy_consumed += energy_consumed
            trips_today.append((weekdays[week_day], format_time(t_dep), f"{soc_start:.2f}", format_time(t_arr), f"{soc_end:.2f}", f"{non_commute_dist:.2f}", round(non_commute_travel_time), f"{energy_consumed:.2f}", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}", f"{ev.battery_size:.10f}"))
            current_soc = soc_end

        if not trips_today:
            trips_today.append((weekdays[week_day], "No trips", f"{ev.max_soc * ev.battery_size:.2f}", "", f"{ev.max_soc * ev.battery_size:.2f}", "", "", "0.00", f"{total_energy_consumed:.2f}", f"{total_distance:.2f}"))

        trip_data.append((day + 1, trips_today))
        ev.degrade()
        day+=1

    return trip_data


def format_time(time_float):
    """Converts time in float format to HH:MM format"""
    hours = int(time_float)
    minutes = int((time_float - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def sample_commute_times(args):
    """Sample departure and arrival times for commuting, ensuring consistency"""
    t_dep_hour = random.uniform(args.C_dept - 0.25, args.C_dept + 0.25)  # Departure time with small variance
    t_arr_hour = random.uniform(args.C_arr - 0.25, args.C_arr + 0.25)  # Arrival time with small variance
    
    while t_arr_hour <= t_dep_hour:
        t_arr_hour = random.uniform(args.C_arr - 0.25, args.C_arr + 0.25)

    return t_dep_hour, t_arr_hour

def sample_non_commute_times(args):
    """Randomly sample departure and arrival times for non-commuting trips, ensuring consistency"""
    t_dep_hour = random.uniform(8, 20)  # Assuming non-commuting trips can start between 8 AM and 8 PM
    trip_duration = random.uniform(0.5, 2)  # Duration between 30 minutes and 2 hours
    t_arr_hour = t_dep_hour + trip_duration

    if t_arr_hour >= 24:
        t_arr_hour -= 24

    return t_dep_hour, t_arr_hour

def write_to_csv(file_name, data):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Day", "Weekday", "Departure Time", "SOC on Departure", "Arrival Time", "SOC on Arrival", "Distance (km)", "Travel Time (min)", "Energy Consumed (kWh)", "Total Energy Consumed (kWh)", "Total Distance (km)", "Battery Size"])
        for day, trips in data:
            for trip in trips:
                writer.writerow([day, *trip])


def run_simulation(args):
    validate_input(args)
    ev = ElectricVehicle(args.ev_battery, args.max_soc, args.min_soc, args.consumption)
    return generate_trip_data(args, ev)


def main(args):
    validate_input(args)
    ev = ElectricVehicle(args.ev_battery, args.max_soc, args.min_soc, args.consumption)
    trip_data = generate_trip_data(args, ev)
    write_to_csv(args.output, trip_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample synthetic EV usage data.')

    # Adding all the arguments from your initial code
    parser.add_argument('--output', type=str, default='ev_usage.csv', help='Output file name')
    #parser.add_argument('--days', type=int, default=365, help='Desired number of days sampled')
    parser.add_argument('--ev_battery', type=float, default=40, help='EV battery size in kWh')
    parser.add_argument('--max_soc', type=float, default=0.8, help='Maximum state of charge')
    parser.add_argument('--min_soc', type=float, default=0.2, help='Minimum state of charge')
    parser.add_argument('--consumption', type=float, default=164, help='Consumption in Wh/km')
    parser.add_argument('--wfh_monday', type=int, default=0, choices=[0, 1], help='WFH on Monday')
    parser.add_argument('--wfh_tuesday', type=int, default=0, choices=[0, 1], help='WFH on Tuesday')
    parser.add_argument('--wfh_wednesday', type=int, default=0, choices=[0, 1], help='WFH on Wednesday')
    parser.add_argument('--wfh_thursday', type=int, default=0, choices=[0, 1], help='WFH on Thursday')
    parser.add_argument('--wfh_friday', type=int, default=0, choices=[0, 1], help='WFH on Friday')
    parser.add_argument('--C_dist', type=float, default=20.0, help='Typical one-way commute distance in km')
    parser.add_argument('--C_dept', type=float, default=7.45, help='Departure time for commuting')
    parser.add_argument('--C_arr', type=float, default=17.30, help='Arrival time from commuting')
    parser.add_argument('--N_nc', type=int, default=5, help='Weekly number of non-commuting one-way trips')


    args = parser.parse_args()
    try:
        main(args)
    except ValueError as e:
        print(f"Input Error: {e}")