import test
import random
import time
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
import pickle
import os

# Helper function to get alpha based on p1j (hours)
def get_alpha_for_demand(p1j_hours, total_N_cars_in_instance, current_car_idx):
    """
    Determines alpha based on p1j (charging time on type 1 charger)
    and special conditions for N_cars = 200 (Group 6 from paper).
    """
    # Group 6: N_cars = 200, specific alpha values
    if total_N_cars_in_instance == 200:
        if current_car_idx < total_N_cars_in_instance / 2:
            return 0.1
        else:
            return 0.2

    # Table 3 from the paper for other N_cars values
    if 0.5 <= p1j_hours < 1.0:
        return random.uniform(0.1, 1.0)
    elif 1.0 <= p1j_hours < 2.0:
        return random.uniform(0.1, 0.9)
    elif 2.0 <= p1j_hours < 3.0:
        return random.uniform(0.1, 0.8)
    elif 3.0 <= p1j_hours < 4.0:
        return random.uniform(0.1, 0.7)
    elif 4.0 <= p1j_hours < 5.0:
        return random.uniform(0.1, 0.6)
    elif p1j_hours >= 5.0:
        return random.uniform(0.1, 0.5)
    else:
        return random.uniform(0.1, 1.0)

def generate_instance(N_cars, SLOT_DURATION_HOURS=1.0):
    """
    Generates a problem instance based on the paper's methodology.

    Args:
        N_cars (int): Number of charging demands (vehicles).
        SLOT_DURATION_HOURS (float): Duration of a single time slot in hours.

    Returns:
        tuple: (J, simulation_start_time_hours, H, charger_types_kW, N_physical_chargers)
            J: List of ChargingDemand objects.
            simulation_start_time_hours: The earliest vehicle arrival time in hours (this is t_0).
            H: List of time slot indices [0, 1, ..., T-1].
            charger_types_kW: List of charger power ratings in kW [11.0, 22.0, 43.0].
    """

    charger_types_kW = [11.0, 22.0, 43.0]  # As per paper
    w_slowest_charger_kW = min(charger_types_kW) # Type 1 charger power

    demands_in_hours = [] # Store (arrival_h, departure_h, energy_kWh)

    for i in range(N_cars):
        # 1. Arrival time (rj) in hours: U[0, 0.2*N_cars]
        arrival_h = random.uniform(0, 0.2 * N_cars)
        arrival_h = int(arrival_h)

        # 2. Required energy (ej) in kWh: U[5.5, 66]
        energy_kWh = random.uniform(5.5, 66.0)
        energy_kWh = int(energy_kWh)

        # 3. Calculate p1j (charging time on type 1 charger)
        p1j_hours = energy_kWh / w_slowest_charger_kW
        p1j_hours = math.ceil(p1j_hours)

        # 4. Determine alpha
        alpha = get_alpha_for_demand(p1j_hours, N_cars, i)

        # 5. Calculate departure time (dj) in hours
        parking_duration_h = (1 + alpha) * p1j_hours
        parking_duration_h = math.ceil(parking_duration_h)
        departure_h = arrival_h + parking_duration_h
        
        demands_in_hours.append({'r_h': arrival_h, 'd_h': departure_h, 'e_kWh': energy_kWh})

    # Determine overall time frame from continuous hour-based demands
    min_overall_arrival_h = min(d['r_h'] for d in demands_in_hours)

    J_slot_based = [] # Final list of ChargingDemand objects

    for demand_info in demands_in_hours:
        # Normalize times relative to the first arrival and discretize
        # r_slot: arrival time slot index (floor)
        # d_slot: departure time slot index (ceil, must be > r_slot)
        r_slot_idx = math.floor((demand_info['r_h'] - min_overall_arrival_h) / SLOT_DURATION_HOURS)
        d_slot_idx = math.ceil((demand_info['d_h'] - min_overall_arrival_h) / SLOT_DURATION_HOURS)

        # Ensure departure is at least one slot after arrival
        if d_slot_idx <= r_slot_idx:
            d_slot_idx = r_slot_idx + 1
        
        # Ensure minimum parking duration constraint from your original code (if applicable in slots)
        # Original: if d - r < 3: continue
        # Let's say min duration is 1 slot after the above adjustment.
        # If you need a longer minimum duration in slots (e.g., 3 slots):
        MIN_SLOT_DURATION = 1 # Or 3, if that's a hard constraint for your model
        if d_slot_idx - r_slot_idx < MIN_SLOT_DURATION:
             # This might happen if SLOT_DURATION_HOURS is very large compared to parking_duration_h
             # Option 1: Adjust d_slot_idx
             d_slot_idx = r_slot_idx + MIN_SLOT_DURATION
             # Option 2: Skip this demand (like your 'continue'). This might reduce N_cars.
             # continue
        
        cd = test.ChargingDemand(r_slot_idx, d_slot_idx, demand_info['e_kWh'])
        J_slot_based.append(cd)
    
    if not J_slot_based: # If all demands were skipped by some filter
        # Fallback to avoid errors if J becomes empty, though unlikely with current logic
        max_demand_departure_slot = 0
    else:
        max_demand_departure_slot = max(cd.d for cd in J_slot_based) if J_slot_based else 0
    
    # H: Time slots from 0 up to the max departure slot needed.
    # If max_demand_departure_slot is, e.g., 10 (meaning car leaves *at the start* of slot 10,
    # so it occupies up to slot 9), then H should be range(10).
    H = list(range(max_demand_departure_slot))

    # The 'beginning_time_slot' your MGCDC expects could be this min_overall_arrival_h
    # if it needs to know the "absolute" start time of slot 0.
    simulation_start_time_hours = min_overall_arrival_h
    
    return J_slot_based, simulation_start_time_hours, H, charger_types_kW

if __name__ == "__main__":
    times = []
    gurobi_failed = False
    
    multipliers = [3,4,5,6,7,8,9,10,15,20,30,40,50]
    
    if os.path.exists("times.pkl"):
        with open("times.pkl", "rb") as f:
            times = pickle.load(f)
    
    already_done = []
    for i in range(len(times)):
        already_done.append((times[i]["num_cars"], times[i]["num_chargers"]))
    
    for num_cars in tqdm(multipliers):
        for num_chargers in [int(m/2) for m in multipliers]:
            
            if (num_cars, num_chargers) in already_done:
                continue
            
            J, beginning_time_slot, H, charger_types = generate_instance(num_cars)
                        
            try:
                MGCDC = test.MinimizingGridCapacityDifferentChargers(J, H, beginning_time_slot, charger_types)
                start = time.time()
                m, demands, charger_powers = MGCDC.minimum_number_of_diverse_chargers()
                optimal_grid_capacity, x_matrix, y_matrix, z_matrix = MGCDC.solve()
                end = time.time()
                
            except Exception as e:
                print(e)
                start = None
                end = None
                optimal_grid_capacity = None
            
            MGCDCH = test.MinimizingGridCapacityDifferentChargersHeuristics(J, H, beginning_time_slot, charger_types)
            MGCDCH2 = test.MinimizingGridCapacityDifferentChargersHeuristics2(J, H, beginning_time_slot, charger_types, 
                                                                            optimal_grid_capacity, optimal_grid_capacity*0.2)

            start3 = time.time()
            wG, wgT, sigmas, schedule, bs, wG_bs = MGCDCH.heuristic_grid_capacity_minimization(J, len(H))
            end3 = time.time()

            initial_solution = test.Solution(J, sigmas, schedule, beginning_time_slot, charger_types)
            # ILS parameters:
            pert0 = 1 # how many times is perturbation performed
            pertMax = 50
            iterMax = 5 # maximum number of non-improving iterations before increasing perturbation
            r = 0.1 # reducing factor, the higher the more likely to accept non-improving solutions
            # SA parameters
            max_generated = 10 # max number of solutions evaluated at same temperature
            acceptance_ratio = 1 # max number of accepted solution at same temperature
            final_temperature = 1
            max_trials = 100 # max number of trials across all temperatures
            mu_temperature = 0.9 # temperature coefficient. The higher, the higher the starting temperature
            
            start2 = time.time()
            solution, optimal_n_max, optimal_ls_max = MGCDCH2.iterated_local_search(initial_solution, 
                                                                                    1000 if num_cars < 10 else 10000, 
                                                                                    1000 if num_chargers < 10 else 10000)
            end2 = time.time()
            
            print("=========================")
            print(optimal_n_max, optimal_ls_max)
            instance = (J, beginning_time_slot, H, charger_types)
            times.append({
                "o": end-start,
                "h": end2-start2,
                "generation": end3-start3,
                "optimal_solution": optimal_grid_capacity,
                "heuristic_solution": solution.wG,
                'num_cars': num_cars,
                'num_chargers': num_chargers,
                'optimal_n_max': optimal_n_max,
                'optimal_ls_max': optimal_ls_max,
                "instance": instance
            })
            
            with open("times.pkl", "wb") as f:
                pickle.dump(times, f)
            
    
    print(times)
    with open("times.pkl", "wb") as f:
        pickle.dump(times, f)