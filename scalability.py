import test
import random
import time
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
import pickle

def generate_instance(N_cars, N_chargers, num_time_slots):
    J = []
    i = 0
    while i < N_cars:
        r = int(random.random() * num_time_slots * 0.2)
        d = r + int(random.random() * num_time_slots * 0.8)
        if d - r < 3: continue
        e = 10 + int(random.random()*50)
        cd = test.ChargingDemand(r, d, e)
        J.append(cd)
        i += 1
    
    beginning_time_slot = min(J, key=lambda x: x.r).r
    
    H = [i for i in range(
        max(J, key=lambda x: x.d).d - beginning_time_slot,
        )] # Time slots

    MGCDC = test.MinimizingGridCapacityDifferentChargers(J, H, beginning_time_slot, [])
    m, demands, charger_powers = MGCDC.minimum_number_of_diverse_chargers()
    if m > N_chargers:
        print("INCREASED NUMBER OF CHARGERS")
    N_chargers = max(N_chargers, m)
    
    e_min = min(J, key=lambda x: x.e).e
    e_max = max(J, key=lambda x: x.e).e
    charger_types = [e_max, e_min]
    for _ in range(N_chargers - 2):
        c = e_min + (e_max - e_min) * random.random()
        c = math.ceil(c)
        charger_types.append(c)
    
    return J, beginning_time_slot, H, charger_types

if __name__ == "__main__":
    times = []
    gurobi_failed = False
    
    for multiplier in tqdm([5,10,15,20,30,40,50]):
        J, beginning_time_slot, H, charger_types = generate_instance(
            multiplier, 
            multiplier, 
            10 if multiplier < 10 else 100)
        
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
                                                                                100 if multiplier < 10 else 1000, 
                                                                                100 if multiplier < 10 else 1000)
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
            'multiplier': multiplier,
            'optimal_n_max': optimal_n_max,
            'optimal_ls_max': optimal_ls_max,
            "instance": instance
        })
        
        with open("times.pkl", "wb") as f:
            pickle.dump(times, f)
            
    
    print(times)
    with open("times.pkl", "wb") as f:
        pickle.dump(times, f)
    # sample:
    # times = [{'o': 0.04794454574584961, 'h': 0.34923481941223145, 'delta': 2.0}, {'o': 0.1030266284942627, 'h': 0.374739408493042, 'delta': 7.0}, {'o': 0.23575353622436523, 'h': 0.3360159397125244, 'delta': 4.0}, {'o': 0.4121570587158203, 'h': 0.3348729610443115, 'delta': 18.0}, {'o': None, 'h': 47.16238760948181, 'delta': None}, {'o': None, 'h': 44.9217529296875, 'delta': None}, {'o': None, 'h': 46.07182168960571, 'delta': None}]

    
    # y1 = []
    # for d in times:
    #     if d["o"] is not None:
    #         y1.append(d["o"])
    # y2 = [d["h"] for d in times]
    # 
    # plt.plot(y1, label="Gurobi time")
    # plt.plot(y2, label="Heuristic time")
    # plt.legend()
    # plt.show()