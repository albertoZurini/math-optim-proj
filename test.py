import numpy as np
import gurobipy as gb
import math
import random

class ChargingDemand:
    def __init__(self, r, d, e):
        """
        :param r: arrival time
        :param d: departure time
        :param e: energy demand
        """
        self.r = r
        self.d = d
        self.e = e
    
    def __repr__(self):
        return f"ChargingDemand(r={self.r}, d={self.d}, e={self.e})"

class Solution:
    def __init__(self, J: list[ChargingDemand], sigmas, schedule, beginning_time_slot, available_charger_types):
        self.J = J
        self.sigmas = sigmas # sigmas[i] tells which charger vehicle j should use
        self.schedule = schedule # aka X
        self.beginning_time_slot = beginning_time_slot
        self.available_charger_types = available_charger_types # array containing powers
        self.H = [i for i in range(
            max(J, key=lambda x: x.d).d - beginning_time_slot,
        )] # Time slots

        self.calculate_wG()
    
    def calculate_wG(self):
        instant_powers = self.sigmas @ self.schedule
        self.wG = max(instant_powers)
    
    def copy(self):
        """Creates a deep copy of the solution for independent modification."""
        # sigmas is assumed to be constant and can be shallow copied (referenced)
        # schedule needs a deep copy as it's modified
        copied_schedule = self.schedule.copy()
        copied_sigmas = self.sigmas.copy()
        return Solution(self.J, copied_sigmas, copied_schedule, self.beginning_time_slot, self.available_charger_types)

    def __repr__(self):
        return f"Solution(wG={self.wG:.4f})"

class MinimizingGridCapacitySameChargers:
    def __init__(self, J: list[ChargingDemand], H: list, beginning_time_slot):
        self.J = J
        self.H = H
        self.beginning_time_slot = beginning_time_slot
        self.num_time_slots = len(self.H)
    
    def minimum_number_of_identical_chargers(self):
        J = sorted(self.J, key=lambda x: x.r)
    
        self.demands = np.zeros((1, self.num_time_slots))
        
        for j_i, j in enumerate(J):
            r = j.r - self.beginning_time_slot
            d = j.d - self.beginning_time_slot
            
            # Check if there's room in this row
            for i in range(self.demands.shape[0]):
                if self.demands[i, r] == 0:
                    self.demands[i, r:d] = j_i+1
                    break
                else:
                    # Add a new row
                    self.demands = np.vstack((self.demands, np.zeros(self.num_time_slots)))
                    self.demands[-1, r:d] = j_i+1
                    break
        
        self.m = self.demands.shape[0]
        
        return self.m
    
    def solve(self, P):
        minimum_grid_capacity = gb.Model()

        x = minimum_grid_capacity.addVars(len(J), len(H), vtype=gb.GRB.BINARY, name="x")
        wG = minimum_grid_capacity.addVar(vtype=gb.GRB.CONTINUOUS, name="wG")

        # Add constraint 2
        for j in range(len(J)):
            minimum_grid_capacity.addConstr(
                gb.quicksum(x[j, t] for t in range(len(H))) == P[j],
                name=f"charging_demand_{j}"
            )
            
        # Add constraint 3
        for t in range(len(H)):
            minimum_grid_capacity.addConstr(
                gb.quicksum(w * x[j, t] for j in range(len(J))) <= wG,
                name=f"total_power_{t}"
            )
            
        # Add constraint 4
        for j in range(len(J)):
            for t in range(len(H)):
                if  t < J[j].r - beginning_time_slot or \
                    t > J[j].d - beginning_time_slot:
                    minimum_grid_capacity.addConstr(
                        x[j, t] == 0,
                        name=f"not_charged_if_not_parked_{j}_{t}"
                    )
                
        minimum_grid_capacity.setObjective(wG, gb.GRB.MINIMIZE)
        
        minimum_grid_capacity.optimize()

        if minimum_grid_capacity.status == gb.GRB.OPTIMAL:
            optim = int(minimum_grid_capacity.objVal)
            return optim
        else:
            raise Exception("Gurobi couldn't find an optimal solution")
    
    def get_max_flow(self, J, m_hat, S_V, intervals, interval_to_i):
        """
        Calculates the maximum flow of energy that can be distributed to a set of vehicles over specified time intervals,
        considering vehicle availability, interval capacities, and maximum power constraints.
        Args:
            J (list): List of vehicle demand objects, each with attributes 'r' (ready time) and 'd' (deadline).
            m_hat (float): Chargers to be activated simultaneously.
            S_V (list): Mapping from source to vehicle (how much energy each vehicle needs).
            intervals (list of tuple): List of time intervals, each represented as a tuple (start_time, end_time).
            interval_to_i (dict): Mapping from interval start time to its index in the intervals list.
        Returns:
            float: The calculated maximum flow of energy that can be delivered to the vehicles within the given constraints.
        """
        
        # Calculate the matrix binding vehicles to intervals
        V_I = np.zeros((len(J), len(intervals)))
        available_capacity = [] # maximum dispensable energy for each time interval
        
        for interval in intervals:
            available_capacity.append(m_hat * (interval[1] - interval[0])) # how many chargers * time interval
        
        for i in range(len(S_V)):
            # for each vehicle demand I have to put how much time this stays plugged in
            residual = S_V[i] # residual energy
            
            interesting_intervals = [] # Intervals when the car is plugged in and can be charged
            for interval in intervals:
                if interval[0] >= J[i].r and interval[1] <= J[i].d:
                    interesting_intervals.append(interval)
            
            for interesting_interval in interesting_intervals:
                if residual == 0:
                    break
                slot_duration = interesting_interval[1] - interesting_interval[0]
                interval_id = interval_to_i[interesting_interval[0]]
                how_much_time = min(residual, slot_duration)
                available_capacity[interval_id] -= how_much_time
                if available_capacity[interval_id] < 0:
                    continue
                V_I[i][interval_id] = how_much_time # time duration of the interval for which the vehicle is charged on the interval
                residual -= how_much_time
        
        # Calculate interval to sink mapping
        I_P = []
        for interval in intervals:
            I_P.append(m_hat * (interval[1] - interval[0]))
        
        # Now I have to calculate the maximum flow of power from the interval to the sink
        max_flow = 0
        for col_i in range(V_I.shape[1]):
            # Sum this column
            col_sum = sum(V_I[:, col_i])
            max_flow += min(col_sum, I_P[col_i])
            
        return max_flow + residual

    
    def max_chargers_activated_simultaneously(self):
        # Calculate source to vehicle mapping
        S_V = []
        for j in self.J:
            S_V.append(j.e/10)
        P = sum(S_V)
        
        # Get each unique time slot
        L = []
        for j in self.J:
            if j.r not in L:
                L.append(j.r)
            if j.d not in L:
                L.append(j.d)
        L.sort()
        
        # Calculate each possible interval
        intervals = []
        interval_to_i = {}
        for i in range(len(L)-1):
            intervals.append((L[i], L[i+1])) # arrival, departure
            interval_to_i[L[i]] = i
        
        # Now I have to do the binary search on this array containing the optimal number of chargers
        m_hats = [ i for i in range(1, self.m+1) ]
        m_hat = -1
        
        i = math.ceil(len(m_hats) / 2)
        max_flow = -1
        while max_flow != P:
            m_hat = m_hats[i]
            max_flow = self.get_max_flow(J, m_hat, S_V, intervals, interval_to_i)
            
            if max_flow < P:
                m_hats = m_hats[i+1:]
            elif max_flow > P:
                m_hats = m_hats[:i]
            else:
                break
            
            i = math.ceil(len(m_hats) / 2)
        
        return m_hat

    
if __name__ == "__main__":
    J = [
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(8, 10, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(9, 12, 30)
    ]

    beginning_time_slot = min(J, key=lambda x: x.r).r
    H = [i for i in range(
        max(J, key=lambda x: x.d).d - beginning_time_slot,
        )] # Time slots
    
    MGCSC = MinimizingGridCapacitySameChargers(J, H, beginning_time_slot)
    m = MGCSC.minimum_number_of_identical_chargers()
    
    assert m == 5
    
    print("Optimal number of chargers", m)
    
    P = [] # Charging time of vehicles
    w = 10 # Power of each charger
    for j in J:
        P.append(j.e/w)
        
    optim = MGCSC.solve(P)
    
    assert optim == 40
    
    print("Optimal grid power", optim)
    
    max_chargers = MGCSC.max_chargers_activated_simultaneously()
    
    assert max_chargers == 4
    
    print("Max chargers activated simultaneously", max_chargers)

# SECOND PART OF THE PAPER: DISTINCT TYPES OF CHARGERS

def required_charging_power(demand: ChargingDemand):
    return demand.e

class MinimizingGridCapacityDifferentChargers:
    def __init__(self, J, H, beginning_time_slot, charger_types):
        self.J = J
        self.H = H
        self.num_time_slots = len(H)
        self.beginning_time_slot = beginning_time_slot
        self.charger_types = charger_types
    
    def minimum_number_of_diverse_chargers(self):
        J = sorted(self.J, key=lambda x: x.r)

        # demands[i, t] = 1 if charger i is busy at time t, 0 if free
        active_chargers_schedules = [] # List of numpy arrays (one per charger)
        active_charger_powers = [] # List of powers for each active charger

        for j_i, j_demand in enumerate(J):
            r = j_demand.r - self.beginning_time_slot
            d = j_demand.d - self.beginning_time_slot
            wlj = required_charging_power(j_demand) # Power needed by current demand
            # assert lj in self.charger_types # the charging power should be available

            # --- Corresponds to Algorithm 2 Line 4: Find Sj ---
            available_chargers_indices = [] # Indices in active_charger_powers
            for i in range(len(active_chargers_schedules)):
                if active_chargers_schedules[i][r] == 0: # Charger i is free at time r
                    available_chargers_indices.append(i)

            # Order available_chargers_indices by their power (non-decreasing)
            # Sj is effectively (index, power) tuples for available chargers
            Sj = sorted(
                [(idx, active_charger_powers[idx]) for idx in available_chargers_indices],
                key=lambda item: item[1] # Sort by power
            )

            found_charger_to_assign = False

            # --- Corresponds to Algorithm 2 Lines 8-10 ---
            if Sj: # If there are available chargers
                suitable_charger_idx_in_Sj = -1
                for k_sj, (charger_original_idx, charger_w_i) in enumerate(Sj):
                    if charger_w_i >= wlj: # Found first suitable charger (wi >= wlj)
                        suitable_charger_idx_in_Sj = k_sj
                        break
                
                if suitable_charger_idx_in_Sj != -1:
                    # --- Algorithm 2 Line 10
                    # Assign the demand to charger i
                    charger_to_use_original_idx = Sj[suitable_charger_idx_in_Sj][0]
                    # Assign demand j to this charger
                    active_chargers_schedules[charger_to_use_original_idx][r:d] = j_i + 1
                    found_charger_to_assign = True
                else:
                    # --- Corresponds to Algorithm 2 Line 11 (No suitable, but available ones exist) ---
                    # All chargers in Sj are too weak (wi < wlj)
                    # "Replace the last charger in Sj" (means highest power one in Sj)
                    charger_to_upgrade_original_idx = Sj[-1][0] # Last in Sj is highest power (but still < lj)
                    
                    # Upgrade its power
                    active_charger_powers[charger_to_upgrade_original_idx] = wlj
                    # Assign demand j to this now-upgraded charger
                    active_chargers_schedules[charger_to_upgrade_original_idx][r:d] = j_i + 1
                    found_charger_to_assign = True
            
            # --- Corresponds to Algorithm 2 Lines 5-7 (Sj was empty OR handled above) ---
            if not found_charger_to_assign: # This means Sj was empty initially
                # Assign demand j to a new charger of type lj
                new_charger_schedule = np.zeros(self.num_time_slots)
                new_charger_schedule[r:d] = j_i + 1
                active_chargers_schedules.append(new_charger_schedule)
                active_charger_powers.append(wlj)
                # m implicitly increments by len(active_charger_powers)

        self.m = len(active_charger_powers)
        
        return self.m, np.array(active_chargers_schedules), active_charger_powers

    def solve(self):
        W = self.charger_types
        P = np.zeros((len(self.J), len(self.charger_types)))
        for j in range(P.shape[0]):
            for l in range(P.shape[1]):
                P[j, l] = math.ceil(self.J[j].e / self.charger_types[l])

        minimum_grid_capacity2 = gb.Model()

        x = minimum_grid_capacity2.addVars(len(self.J), len(self.H), vtype=gb.GRB.BINARY, name="x")
        y = minimum_grid_capacity2.addVars(len(self.charger_types), len(self.J), vtype=gb.GRB.BINARY, name="y")
        z = minimum_grid_capacity2.addVars(len(self.J), len(self.charger_types), len(self.H), vtype=gb.GRB.BINARY, name="z")
        wG = minimum_grid_capacity2.addVar(vtype=gb.GRB.CONTINUOUS, name="wG")

        # Add constraints 8
        for j in range(len(self.J)):
            minimum_grid_capacity2.addConstr(
                gb.quicksum(x[j, t] for t in range(len(self.H))) == gb.quicksum(P[j, l] * y[l, j] for l in range(len(self.charger_types))),
                name=f"charging_demand_{j}"
            )
            
        # Add constraints 9
        for j in range(len(self.J)):
            minimum_grid_capacity2.addConstr(
                gb.quicksum(y[l, j] for l in range(len(self.charger_types))) == 1,
                name=f"charger_type_{j}"
            )

        # Add constraints 11
        for j in range(len(self.J)):
            for t in range(len(self.H)):
                if  t < self.J[j].r - self.beginning_time_slot or \
                    t > self.J[j].d - self.beginning_time_slot:
                    minimum_grid_capacity2.addConstr(
                        x[j, t] == 0,
                        name=f"not_charged_if_not_parked_{j}_{t}"
                    )

        for j in range(len(self.J)):
            for l in range(len(self.charger_types)):
                for t in range(len(self.H)):
                    # Add constraints 12
                    minimum_grid_capacity2.addConstr(
                        z[j, l, t] >= x[j, t] + y[l, j] - 1,
                        name=f"z_geq_{j}_{l}_{t}"
                    )
                    
                    # Add constraints 13
                    minimum_grid_capacity2.addConstr(
                        z[j, l, t] <= x[j, t],
                        name=f"z_leq_x_{j}_{l}_{t}"
                    )
                    
                    # Add constraints 14
                    minimum_grid_capacity2.addConstr(
                        z[j, l, t] <= y[l, j],
                        name=f"z_leq_y_{j}_{l}_{t}"
                    )
                    
        # Add constraints 15
        for t in range(len(self.H)):
            minimum_grid_capacity2.addConstr(
                gb.quicksum(W[l] * z[j, l, t] for j in range(len(self.J)) for l in range(len(self.charger_types))) <= wG,
                name=f"w_{l}_z_{j}_{l}_{t}_leq_wG"
            )

        # Add constraints 16
        # m = []
        # for l in range(len(W)):
        #     minimum_grid_capacity2.addConstr(
        #         gb.quicksum(y[l, j] for j in range(len(J))) <= m[l],
        #         name=f"y_{l}_leq_m_{l}"
        #     )
                
        minimum_grid_capacity2.setObjective(wG, gb.GRB.MINIMIZE)
        
        minimum_grid_capacity2.optimize()

        x_matrix = np.full((len(self.J), len(self.H)), np.nan)
        y_matrix = np.full((len(self.charger_types), len(self.J)), np.nan) # Based on y[l,j] definition
        z_matrix = np.full((len(self.J), len(self.charger_types), len(self.H)), np.nan)
        wG_value = np.nan

        if minimum_grid_capacity2.status == gb.GRB.OPTIMAL:
            optim = int(minimum_grid_capacity2.objVal)
            print("\nOptimal grid capacity:", optim)
            
            wG_value = int(minimum_grid_capacity2.objVal)
            
            x_matrix = np.array(
                [[x[j_idx, t_idx].X for t_idx in range(len(self.H))]
                for j_idx in range(len(self.J))]
            )
            y_matrix = np.array(
                [[y[l_idx, j_idx].X for j_idx in range(len(self.J))]
                for l_idx in range(len(self.charger_types))]
            )
            z_matrix = np.array(
                [[[z[j_idx, l_idx, t_idx].X for t_idx in range(len(self.H))]
                for l_idx in range(len(self.charger_types))]
                for j_idx in range(len(self.J))]
            )
            
            return wG_value, x_matrix, y_matrix, z_matrix
        else:
            raise Exception("Gurobi couldn't find an optimal solution to the problem!")

if __name__ == "__main__":
    J = [
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(8, 10, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(9, 12, 30)
    ]

    beginning_time_slot = min(J, key=lambda x: x.r).r
    H = [i for i in range(
        max(J, key=lambda x: x.d).d - beginning_time_slot,
        )] # Time slots
    charger_types = [10, 20, 30]
    
    MGCDC = MinimizingGridCapacityDifferentChargers(J, H, beginning_time_slot, charger_types)
    
    m, demands, charger_powers = MGCDC.minimum_number_of_diverse_chargers()
    
    ### DATA FROM THE PAPER
    X = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ]) #  who's charging when

    Y = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]) # V[i] is charging on W[j]

    P = np.array([
        [1, 2, 1, 1, 1],
        [1, 3, 2, 2, 2],
        [1, 2, 1, 1, 1],
        [1, 2, 1, 1, 1],
        [1, 2, 1, 1, 1],
        [1, 2, 1, 1, 1]
    ]) # [i][j] how long to charge V[i] on W[j]

    W = np.array([[30, 10, 20, 20, 20]]) # chargers power
    
    ### END DATA FROM PAPER
    
    cp = W @ Y @ X
    cp_comparison = cp == charger_powers

    print("Charging powers vs optimal solution", cp_comparison)
    # Both True and False should appear in this array
    # This happens because the example in the text is an optimal solution
    # The one found is just to find the minimum number of chargers, so it will choose a 20kw rather than a 10kw
    
    optimal_grid_capacity, x_matrix, y_matrix, z_matrix = MGCDC.solve()
    
    assert optimal_grid_capacity == 30
    
    print("Optimal grid capacity", optimal_grid_capacity)

### SOLVE USING HEURISTICS ###

class MinimizingGridCapacityDifferentChargersHeuristics:
    def __init__(self, J, H, beginning_time_slot, charger_types):
        self.J = J
        self.H = H
        self.num_time_slots = len(H)
        self.beginning_time_slot = beginning_time_slot
        self.charger_types = charger_types

    def calculate_lb(self, J):
        # Lower bound for minimum grid capacity
        d = [j.d for j in J]
        r = [j.r for j in J]
        e = [j.e for j in J]
        
        # First term: ceil(total_energy / total_available_time)
        total_energy = sum(e)
        total_time = max(d) - min(r)
        e1 = np.ceil(total_energy / total_time)
        
        # Second term: find minimal charger type >= max(ej/(dj-rj))
        e2_values = [math.ceil(ej / (dj - rj)) for ej, dj, rj in zip(e, d, r) if dj > rj]
        e2 = max(e2_values) if e2_values else 0
        
        # Find smallest charger type >= e2
        sorted_W = sorted(self.charger_types)
        wl = e2 # This is added because with random data the for loop might not work
        for w in sorted_W:
            if w >= e2:
                wl = w
                break
        
        return max(e1, wl)

    def heuristic_grid_capacity_minimization(self, J, num_time_slots):
        # This is algorithm 3
        W = np.array(self.charger_types)
        J = sorted(J, key=lambda x: (x.d, -x.e, x.r))
        
        wgT = np.zeros(num_time_slots) # This is containing the current power delivered to all vehicles for each time slot
        sigmas = np.zeros(len(J)) # This is the charging power assigned to each vehicle
        schedule = np.zeros((len(J), num_time_slots)) # This is when a vehicle is charging
        bs = []
        wG_bs = []
        
        lb = self.calculate_lb(J) # lower bound
        
        wG = lb
        for j_i, j in enumerate(J):
            # First step of the algorithm, is to seek the greatest charging powerrate allowed to charge j without exceeding
            # the current grid capacity
            
            # --- Line 5 
            # first I have to count how many free slots there are
            max_lb_wG = max(lb, wG)  # Current power threshold
            
            # Get time interval [rj, dj)
            start = j.r - self.beginning_time_slot
            end = j.d - self.beginning_time_slot
            interval = wgT[start:end]  # Slicing is [start, end)
                    
            # Calculate available slots (b), where the delivered power is less than the threshold
            mask = interval < max_lb_wG
            b = np.sum(mask)
            
            bs.append(b)
            
            # --- Line 6
            # secondly I have to calculate the maximum power delivered
            
            # Calculate wG_b (max power in available slots)
            if b > 0:
                wG_b = np.max(interval[mask])
            else:
                wG_b = 0  # No available slots
            
            wG_bs.append(wG_b)

            # --- Lines 7-8 ---
            needed_powerrate = int(j.e/b + wG_b)
            if needed_powerrate <= max_lb_wG:
                # if the power I need is less than the threshold, I can use an existing charger that doesn't trip the threshold
                mask = W <= math.ceil(math.ceil(j.e/b)/10)*10
                if sum(mask) == 0:
                    # This shouldn't happen, but happens with random generated data
                    charger_type = required_charging_power(j)
                else:
                    charger_type = np.max(W[mask])
                
                sigmas[j_i] = charger_type
            else:
                # In this case, the grid capacity will be increased, so I just get the smallest charger that can satisfy the need
                sigmas[j_i] = required_charging_power(j) # Wlj
            
            def next_most_powerful_charger(sigma, types):
                types_sorted = sorted(types)
                for t in types_sorted:
                    if t > sigma:
                        return t
                raise Exception("No more powerful charger found!")
            
            # THIS WAS NOT IN THE PAPER
            while j.e / sigmas[j_i] > end-start:
                # Sometimes the heuristic fails and puts a charger that's not powerful enough
                sigmas[j_i] = next_most_powerful_charger(sigmas[j_i], self.charger_types)
            
            # --- Line 9 --- Schedule the charging according to algorithm 4
            wG, wgT, times = self.power_allocation_heuristic(sigmas, j, j_i, start, end, wgT, max_lb_wG, wG)
            
            # --- Line 10 --- use times to build the schedule
            for t in times:
                schedule[j_i][t] = 1
            
        return wG, wgT, sigmas, schedule, bs, wG_bs

    def power_allocation_heuristic(self, sigmas, j, j_i, start, end, wgT, current_max, wG):
        # This is Algorithm 4
        # It starts charging vehicle j on time slots without exceeding the maximum power between wG and lb in chronological order
        # then on time slots with the minium wG_t value
        # --- Line 1
        p = j.e / sigmas[j_i] # number of time slots required to charge j on sigma_j
                
        # --- Lines 2-3
        H1 = [] # set of time slots where the grid capacity is not exceeded
        H2 = [] # set of time slots where the grid capacity is exceeded, sorted in increasing order of wgT
        for t in range(start, end):
            if sigmas[j_i] + wgT[t] < current_max:
                H1.append(t)
            else:
                H2.append(t)
        
        H2.sort(key=lambda x: wgT[x])
        
        times = []
        
        while p > 0:
            if H1:
                t = H1.pop(0)
            elif H2:
                t = H2.pop(0)
            else:
                raise Exception("No available time slots")

            # --- Line 9
            wgT[t] += sigmas[j_i]
            p -= 1
            
            if wgT[t] > wG:
                wG = wgT[t] # update the grid capacity
                
            times.append(t)
                
        return wG, wgT, times
    
    def Perturbation(self, solution_to_perturb: Solution) -> Solution:
        new_solution = solution_to_perturb.copy()
        
        # The perturbation consists of selecting a charging demand j and changing its charger type sigma_j to a new charger type l, 
        # where l != sigma_j and pjl <= dj - rj
        # basically, the perturbation doesn't touch the schedule, only the chargers

        # --- Part 1: random selection ---
        # Select a random demand and assign it to a random type
        random_demand_idx = random.randint(0, len(new_solution.J) - 1)
        
        # Now that I know which one to replace, I have to find a new type that satisfies the constraints:
        # - l != sigma_j
        # - pjl <= dj - rj
        # pjl is the number of time slots needed to charge j on charger l

        # --- Part 2: Changing charger type ---
        possible_chargers = []
        current_charger = new_solution.sigmas[random_demand_idx]
        for c in new_solution.available_charger_types:
            if c != current_charger:
                add = False
                for j in new_solution.J:
                    if j.e/c <= j.d - j.r:
                        add = True
                        break
                
                if add:
                    possible_chargers.append(c)
        
        chosen_charger = random.choice(possible_chargers)
        
        new_solution.sigmas[random_demand_idx] = chosen_charger
        
        # The perturbation is followed by algorithm 4
        wG, wgT, sigmas, schedule, _, _ = self.heuristic_grid_capacity_minimization(new_solution.J, len(new_solution.H))
        
        new_solution.sigmas = sigmas
        new_solution.schedule = schedule
        new_solution.calculate_wG()
        assert wG == new_solution.wG
        
        return new_solution
    
    def generate(self, initial_solution: Solution):
        # This function should just swap the schedule of a charging car
        new_solution = initial_solution.copy()
        
        def swap_items_in_row(arr):
            arr_copy = arr.copy()

            indices_of_zeros = np.where(arr_copy == 0)[0]
            indices_of_ones = np.where(arr_copy == 1)[0]

            # Check if both 0s and 1s exist in the array
            if len(indices_of_zeros) == 0 or len(indices_of_ones) == 0:
                return arr_copy # Return the copy of the original

            # Randomly select one index for a 0 and one index for a 1
            idx_zero_to_swap = np.random.choice(indices_of_zeros)
            idx_one_to_swap = np.random.choice(indices_of_ones)

            # Perform the swap
            arr_copy[idx_zero_to_swap] = 1
            arr_copy[idx_one_to_swap] = 0
            
            return arr_copy
        
        random_demand_idx = random.randint(0, len(new_solution.J) - 1)
        selected_row = new_solution.schedule[random_demand_idx]
        # Now that we have the row, we have to swap some 0 and 1 while the car is parked
        beginning = new_solution.J[random_demand_idx].r - new_solution.beginning_time_slot
        end = new_solution.J[random_demand_idx].d - new_solution.beginning_time_slot
        selected_row_roi = selected_row[beginning:end]
        
        selected_row_roi = swap_items_in_row(selected_row_roi)
        selected_row[beginning:end] = selected_row_roi
        new_solution.schedule[random_demand_idx] = selected_row
        
        return new_solution

    def LocalSearch(self,
                    current_solution: Solution,
                    max_generated, # maximum number of generated solutions in one iteration
                    acceptance_ratio, # ratio on the maximum number of generated solutions
                    final_temperature, # final temperature for simulated annealing
                    max_trials, # maximum number of trials before stopping
                    mu_temperature # initial temperature coefficient
                    ) -> Solution:
        # This algorithm implements simulated annealing, Algorithm 6
        
        # --- Line 1
        S_best = current_solution.copy()
        S = current_solution.copy()
        T = mu_temperature*current_solution.wG
        M = max_trials/max_generated
        trial = 0
        # b = (T-final_temperature) / (T * M * final_temperature) # Lundy-Mees parameter, it makes the temperature to cool down in M steps
        b = (1/final_temperature - 1/T) / M
        
        # --- Line 2
        max_accepted = acceptance_ratio * max_generated
        
        while True:
            accepted = 0
            generated = 0
            
            while generated <= max_generated and accepted < max_accepted:
                # --- Line 6
                S_prime = self.generate(S)
                
                delta_f = S_prime.wG - S.wG
                
                generated += 1
                trial += 1
                
                u = random.uniform(0, 1)
                acceptance = np.exp(-delta_f/T)
                
                if S_prime.wG < S.wG or u < acceptance:
                    # If the solution doesn't change, the SA will 
                    # move freely in the space of solutions with 
                    # same wG
                    accepted += 1
                    S = S_prime.copy()
                    if S.wG < S_best.wG:
                        S_best = S.copy()
            
            T = T / (1 + b*T)
            
            if not (trial <= max_trials and accepted > 0):
                break            
        
        return S_best
    
    def iterated_local_search(
        self,
        initial_solution: Solution,
        pert0: float,       # Initial perturbation strength/level
        pert_max_abs: float, # Absolute maximum perturbation strength/level
        iter_max: int,      # Max iterations without improvement before increasing pert
        r_accept: float,    # Parameter for acceptance probability of non-improving solutions
        p0_accept: float, # Acceptance probability base for non-improving solutions,
        max_generated,
        acceptance_ratio,
        final_temperature,
        max_trials,
        mu_temperature
    ) -> Solution:
        """
        Implements the Iterated Local Search algorithm (Algorithm 5).
        Assumes f(S) means solution.wG and lower is better.
        """
        iter_count = 0
        current_pert_strength = pert0

        # S, S', S* are Solution objects
        # S = initial_solution.copy() # S_perturbed, S_after_ls
        S_prime = initial_solution.copy()  # S' in pseudocode (base for perturbation)
        S_star = initial_solution.copy()   # S* in pseudocode (best solution found)

        print(f"ILS Start: S0.wG={initial_solution.wG:.4f}, pert0={pert0}, pert_max_abs={pert_max_abs}, iter_max={iter_max}, r={r_accept}")

        main_loop_count = 0
        while current_pert_strength < pert_max_abs:
            main_loop_count += 1
            # print(f"\nILS Main Loop {main_loop_count}: current_pert_strength={current_pert_strength:.2f}, S_star.wG={S_star.wG:.4f}")

            # Line 4: Choose p, number of perturbation steps
            max_p_val = int(round(current_pert_strength))
            if max_p_val < 1:
                max_p_val = 1 # Apply at least one perturbation if pert_strength is very small but > 0
            
            p_perturb_steps = random.randint(1, max_p_val)
            # print(f"  Perturbing {p_perturb_steps} times (max_p_val from strength {current_pert_strength:.2f})")

            # Lines 5-7: Apply Perturbation p times starting from S_prime
            S_perturbed = S_prime.copy() # Start perturbation from S_prime
            for _ in range(p_perturb_steps):
                S_perturbed = self.Perturbation(S_perturbed)
            # print(f"  After Perturbation: S_perturbed.wG={S_perturbed.wG:.4f}")

            # Line 8: Apply Local Search
            S_after_ls = self.LocalSearch(S_perturbed,
                                    max_generated,
                                    acceptance_ratio,
                                    final_temperature,
                                    max_trials,
                                    mu_temperature)
            # print(f"  After LocalSearch: S_after_ls.wG={S_after_ls.wG:.4f}")

            # Lines 9-11: Acceptance Criterion
            if S_after_ls.wG < S_star.wG:
                print(f"  New best found! S_after_ls.wG ({S_after_ls.wG:.4f}) < S_star.wG ({S_star.wG:.4f})")
                S_prime = S_after_ls.copy()
                S_star = S_after_ls.copy()
                iter_count = 0
                current_pert_strength = pert0 # Reset perturbation strength
            else:
                # Lines 12-15: Probabilistic acceptance for non-improving solution
                u = random.uniform(0, 1)
                acceptance_prob = p0_accept * (r_accept ** (iter_count - 1))

                if u < acceptance_prob:
                    # print(f"  Accepted non-improving solution: S_after_ls.wG ({S_after_ls.wG:.4f}), u={u:.2f} < prob={acceptance_prob:.2f}")
                    S_prime = S_after_ls.copy()
                    iter_count += 1
                else:
                    pass
                    # print(f"  Rejected non-improving solution: S_after_ls.wG ({S_after_ls.wG:.4f}), u={u:.2f} >= prob={acceptance_prob:.2f}")

            # Lines 17-19: Increase perturbation strength if stuck
            if iter_count >= iter_max:
                print(f"  iter_count ({iter_count}) >= iter_max ({iter_max}). Increasing perturbation.")
                iter_count = 0
                current_pert_strength += pert0
                S_prime = S_star.copy() # Restart perturbation from the best known solution

        print(f"ILS Finished. Best solution S_star.wG={S_star.wG:.4f}")
        return S_star

## IMPORTANT INFORMATION: the heuristic provided by the paper doesn't converge

##if __name__ == "__main__":
##    J = [
##        ChargingDemand(10, 13, 20),
##        ChargingDemand(10, 13, 20),
##        ChargingDemand(8, 10, 20),
##        ChargingDemand(10, 13, 20),
##        ChargingDemand(10, 13, 20),
##        ChargingDemand(9, 12, 30)
##    ]
##
##    beginning_time_slot = min(J, key=lambda x: x.r).r
##    H = [i for i in range(
##        max(J, key=lambda x: x.d).d - beginning_time_slot,
##        )] # Time slots
##    charger_types = [10, 20, 30]
##    
##    MGCDCH = MinimizingGridCapacityDifferentChargersHeuristics(J, H, beginning_time_slot, charger_types)
##    
##    wG, wgT, sigmas, schedule, bs, wG_bs = MGCDCH.heuristic_grid_capacity_minimization(J, len(H))
##    
##    initial_solution = Solution(J, sigmas, schedule, beginning_time_slot, charger_types)
##    pert0 = 1 # how many times is perturbation performed
##    pertMax = 10
##    iterMax = 5 # maximum number of non-improving iterations before increasing perturbation
##    r = 0.1 # reducing factor, the higher the more likely to accept non-improving solutions
##    solution = MGCDCH.iterated_local_search(initial_solution, pert0, pertMax, iterMax, r, 1,
##                                            max_generated=10,
##                                            acceptance_ratio=0.1,
##                                            final_temperature=2,
##                                            max_trials=10,
##                                            mu_temperature=10)
##    print("Initial solution:", initial_solution.wG)
##    print("New solution:", solution.wG)
    
    
class MinimizingGridCapacityDifferentChargersHeuristics2:
    def __init__(self, J, H, beginning_time_slot, charger_types, optimal=None, margin=None):
        self.J = J
        self.H = H
        self.num_time_slots = len(H)
        self.beginning_time_slot = beginning_time_slot
        self.charger_types = charger_types
        self.optimal = optimal
        self.margin = margin
    
    def mutate(self, solution_to_mutate: Solution) -> Solution:
        S_prime = solution_to_mutate.copy()
        # The mutation consists of selecting a charging demand j and changing its charger type sigma_j to a new charger type l,
        
        which_demand = random.randint(0, len(S_prime.J) - 1)
        current_charger = S_prime.sigmas[which_demand]
        minimum_required_power = S_prime.J[which_demand].e / (S_prime.J[which_demand].d - S_prime.J[which_demand].r)
        # The possible charger should satisfy power constraints and be different
        available_chargers = [c for c in S_prime.available_charger_types if c != current_charger and c >= minimum_required_power]
        which_charger = random.choice(available_chargers)
        S_prime.sigmas[which_demand] = which_charger
        
        # Now I have to recalculate the schedule, based on the new charger
        # Basically I need to activate the charger in the beginning for at least how_many_time_slots_to_charge time units
        how_many_time_slots_to_charge = math.ceil(S_prime.J[which_demand].e / which_charger)
        row = S_prime.schedule[which_demand]
        beginning = S_prime.J[which_demand].r - S_prime.beginning_time_slot
        end = S_prime.J[which_demand].d - S_prime.beginning_time_slot
        row_roi = row[beginning:end]
        # In row_roi there should be how_many_time_slots_to_charge 1s, and the rest 0s
        for i in range(len(row_roi)):
            if i >= how_many_time_slots_to_charge:
                row_roi[i] = 0
            else:
                row_roi[i] = 1
        
        # Just checking that the whole amount of energy required by the vehicle can be satisfied by
        # the charger in a sufficient number of time slot
        assert sum(row_roi) == how_many_time_slots_to_charge
            
        # Update schedule
        S_prime.schedule[which_demand][beginning:end] = row_roi
        
        return S_prime, which_demand
    
    def local_search(self, current_solution: Solution, which_demand) -> Solution:
        def swap_items_in_row(arr):
            arr_copy = arr.copy()
            
            indices_of_zeros = np.where(arr_copy == 0)[0]
            indices_of_ones = np.where(arr_copy == 1)[0]
            
            # Check if both 0s and 1s exist in the arrayAdd commentMore actions
            if len(indices_of_zeros) == 0 or len(indices_of_ones) == 0:
                return arr_copy # Return the copy of the originalAdd comment

            # Randomly select one index for a 0 and one index for a 1
            idx_zero_to_swap = np.random.choice(indices_of_zeros)
            idx_one_to_swap = np.random.choice(indices_of_ones)

            # Perform the swap
            arr_copy[idx_zero_to_swap] = 1
            arr_copy[idx_one_to_swap] = 0
            
            return arr_copy
        
        S_prime = current_solution.copy()
        
        row = S_prime.schedule[which_demand]
        beginning = S_prime.J[which_demand].r - S_prime.beginning_time_slot
        end = S_prime.J[which_demand].d - S_prime.beginning_time_slot
        row_roi = row[beginning:end]
        row_roi2 = swap_items_in_row(row_roi)
        
        S_prime.schedule[which_demand][beginning:end] = row_roi2
        
        return S_prime
    
    def iterated_local_search(self, initial_solution: Solution, n_max, ls_max) -> Solution:
        # The two things that can be changed are the charger type and the schedule
        S = initial_solution.copy()
        S_prime = initial_solution.copy()
        S_star = initial_solution.copy()
        
        optimal_n_max = 0
        optimal_ls_max = 0
        
        optim_found = False
        # How many chargers should be replaced?
        j = 0
        while j < n_max:
            S_prime, index = self.mutate(S)
            
            for i in range(ls_max):
                if i == 0:
                    which_demand = index
                else:
                    which_demand = random.randint(0, len(S_prime.J) - 1)

                S_prime = self.local_search(S_prime, which_demand)
                S_prime.calculate_wG()
                
                if S_prime.wG < S_star.wG:
                    S_star = S_prime.copy()
                    S = S_star.copy()
                    
                    S_star.calculate_wG()
                    
                    if self.optimal is not None and abs(S_star.wG - self.optimal) < self.margin:
                        # Optimal solution found
                        optimal_n_max = j
                        optimal_ls_max = i
                        j = n_max
                        optim_found = True
                        break
                else:
                    # Exploration
                    if random.random() < 0.1: # 10% chance to accept a worse solution
                        S = S_prime.copy()
                    
            j += 1
                
        if not optim_found:
            optimal_n_max = j
            optimal_ls_max = i
 
        return S_star, optimal_n_max, optimal_ls_max

if __name__ == "__main__":
    J = [
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(8, 10, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(10, 13, 20),
        ChargingDemand(9, 12, 30)
    ]
    J = sorted(J, key=lambda x: (x.d, -x.e, x.r))

    beginning_time_slot = min(J, key=lambda x: x.r).r
    H = [i for i in range(
        max(J, key=lambda x: x.d).d - beginning_time_slot,
        )] # Time slots
    charger_types = [10, 20, 30]
    
    MGCDCH = MinimizingGridCapacityDifferentChargersHeuristics(J, H, beginning_time_slot, charger_types)
    MGCDCH2 = MinimizingGridCapacityDifferentChargersHeuristics2(J, H, beginning_time_slot, charger_types)
    
    # Get an initial solution using the heuristic
    wG, wgT, sigmas, schedule, bs, wG_bs = MGCDCH.heuristic_grid_capacity_minimization(J, len(H))
    initial_solution = Solution(J, sigmas, schedule, beginning_time_slot, charger_types)
    
    solution, optimal_n_max, optimal_ls_max = MGCDCH2.iterated_local_search(initial_solution, 1000, 100)
    print("Initial solution:", initial_solution.wG)
    print("New solution:", solution.wG)
    pass