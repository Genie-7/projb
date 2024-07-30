import random
from collections import Counter

# Work pattern distributions by year
WORK_PATTERNS = {
    2017: {"Remote": 0.041, "Hybrid": 0.01, "In-Person": 0.949},
    2018: {"Remote": 0.049, "Hybrid": 0.011, "In-Person": 0.94},
    2019: {"Remote": 0.05, "Hybrid": 0.02, "In-Person": 0.93},
    2020: {"Remote": 0.258, "Hybrid": 0.036, "In-Person": 0.706},
    2021: {"Remote": 0.272, "Hybrid": 0.036, "In-Person": 0.692},
    2022: {"Remote": 0.185, "Hybrid": 0.071, "In-Person": 0.744},
    2023: {"Remote": 0.141, "Hybrid": 0.103, "In-Person": 0.756}
}

# New BEV registrations by year
NEW_BEV_REGISTRATIONS = {
    2017: 9079,
    2018: 22570,
    2019: 35523,
    2020: 39036,
    2021: 58726,
    2022: 98589,
    2023: 139521
}

class ElectricVehicle:
    def __init__(self, model_year, commuter_type, battery_replacement_threshold):
        self.model_year = model_year
        self.commuter_type = commuter_type
        self.battery_replacement_threshold = battery_replacement_threshold
        self.battery_soh = 1.0
        self.age = 0
        self.battery_replacements = 0

    def simulate_year(self):
        self.age += 1
        degradation_rate = self.get_degradation_rate()
        self.battery_soh *= (1 - degradation_rate)
        
        if self.battery_soh <= self.battery_replacement_threshold:
            self.replace_battery()
            return True
        return False

    def get_degradation_rate(self):
        base_rate = 0.03  # 3% base degradation per year
        usage_factor = {"Remote": 0.8, "Hybrid": 1.0, "In-Person": 1.2}
        return base_rate * usage_factor[self.commuter_type]

    def replace_battery(self):
        self.battery_soh = 1.0
        self.battery_replacements += 1

def run_simulation(start_year, end_year, final_year):
    fleet = []
    for year in range(start_year, end_year + 1):
        new_vehicles = []
        for _ in range(NEW_BEV_REGISTRATIONS[year]):
            commuter_type = random.choices(list(WORK_PATTERNS[year].keys()), 
                                           weights=list(WORK_PATTERNS[year].values()))[0]
            battery_threshold = random.choices([0.8, 0.7, 0.6], weights=[0.1, 0.2, 0.7])[0]
            new_vehicles.append(ElectricVehicle(year, commuter_type, battery_threshold))
        fleet.extend(new_vehicles)

    total_replacements = 0
    for current_year in range(start_year, final_year + 1):
        year_replacements = sum(ev.simulate_year() for ev in fleet)
        total_replacements += year_replacements
        
        print(f"\nYear {current_year}:")
        print(f"Total vehicles: {len(fleet)}")
        print(f"Battery replacements this year: {year_replacements}")
        print(f"Total battery replacements: {total_replacements}")

        commuter_counts = Counter(ev.commuter_type for ev in fleet)
        print("Commuter Type Distribution:")
        for t, count in commuter_counts.items():
            print(f"{t}: {count} ({count/len(fleet)*100:.1f}%)")

    avg_lifespan = sum(ev.age for ev in fleet) / len(fleet)
    print(f"\nAverage battery lifespan: {avg_lifespan:.2f} years")
    print(f"Total battery replacements: {total_replacements}")

if __name__ == "__main__":
    run_simulation(2017, 2023, 2048)