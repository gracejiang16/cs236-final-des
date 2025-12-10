from __future__ import annotations
import heapq
import itertools
import json
import math
import random
from typing import Any, Optional, Tuple
from helpers import read_graph, prepopulate_num_cars_at_t
import csv
# import pandas as pd
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
import ast

"""
simple_discrete_event_sim.py

A minimal discrete-event simulator where scheduled events carry an
event_id and optional payload. Event behavior is implemented by
overriding the Simulator.handle(event_id, payload) method using a
simple switch (if/elif) inside it.
"""

class EventHandle:
    """Simple cancelable handle for a scheduled event."""
    __slots__ = ("_cancelled",)
    def __init__(self) -> None:
        self._cancelled = False
    def cancel(self) -> None:
        self._cancelled = True
    @property
    def cancelled(self) -> bool:
        return self._cancelled

class BaselineSimulator:
    """Minimal discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, graph, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0
        self.graph = graph
        self.num_cars_on_edge_at_t = prepopulate_num_cars_at_t(graph)
        self.initial_dijkstra_paths = {} # tracks our intially calcuated paths for each agents
        self.detailed_path_record = {} # see paper for description
        self.end_times = {} # agent_id: time it reached its destination node
        
        self.available_tasks = [] # format is list of 5-tuples (task-id, location (vertex-id), appear-time, target-time, reward)
        self.available_dashers = [] # list of 3-tuple dashers
        self.total_system_score = 0

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        current_time = self.now

        if event_id == 'T': # TaskArrival
            task_id = payload['task id']
            location = payload['location']
            appear_time = payload['appear time']
            target_time = payload['target time']
            reward = payload['reward']

            self.available_tasks.append((task_id, location, appear_time, target_time, reward))

        elif event_id == 'D': # DasherArrival
            start_location = payload['start location']
            dasher_start_time = payload['start time']
            dasher_exit_time = payload['exit time']

            self.available_dashers.append((start_location, dasher_start_time, dasher_exit_time))

            # print(f'before: {self.available_dashers=}')
            feasible_task_index = -1
            smallest_time = float('inf')
            projected_times = graph.dijkstra_shortest_path(str(start_location))
            for i, potential_task in enumerate(self.available_tasks):
                task_location = potential_task[1] # index 1 is task's location
                # print(f'{task_location=}, {projected_time=}')
                specific_time = projected_times[str(task_location)] # finding the time to each reward
                task_target_time = potential_task[3] # index 3 is task's target time
                if (dasher_start_time + specific_time <= task_target_time)\
                    and (dasher_start_time + specific_time <= dasher_exit_time)\
                    and (specific_time < smallest_time):
                    smallest_time = specific_time
                    feasible_task_index = i
            
            if not feasible_task_index == -1:
                try:
                    # mark dasher as unavailable
                    self.available_dashers.remove((start_location, dasher_start_time, dasher_exit_time))
                except Exception:
                    print('dasher not available')

                new_start_location = self.available_tasks[feasible_task_index][1] # dasher's new location is old task's location
                new_dasher_start_time = self.available_tasks[feasible_task_index][3] # dasher's new start time is old task's target time
                dasher_exit_time = payload['exit time'] # dasher's exit time stays same

                # Schedule the next DasherArrival event
                self.schedule_at(new_dasher_start_time, "D", {'start location': new_start_location, 'start time': new_dasher_start_time, 'exit time': dasher_exit_time})

                # Add task's reward to the total-system-score
                self.total_system_score += self.available_tasks[feasible_task_index][4]
                # print(f'{self.total_system_score=}')

                # mark task as completed by removing it from available_tasks list
                self.available_tasks.pop(feasible_task_index)

                # print(f'after: {self.available_dashers=}')
                
            else:
                # feasible task not found; do nothing
                pass

    def final_results_string(self):
        """
        Function to calculate final statistics and make correct format for output printing.
        Only invoke this function after the simulator is done running.
        """
        total_congestion = self.calculate_total_congestion()
        average_congestion = total_congestion / len(self.detailed_path_record)

        output_string = ""

        # now construct strings of this format: "Car 1 (start_node, end_node), arrived at t=.., with path (e1,t1), (e2,t2), ..."
        for agent_id in range(len(self.detailed_path_record)):
            if agent_id not in self.detailed_path_record:
                # if there's no possible path for this agent
                continue

            path = self.detailed_path_record[agent_id]
            current_car_summary = '\n'
            start_node = path[0][0][0]
            end_node = path[-1][0][1]
            arrival_time = self.end_times[agent_id]
            formatted_path = [edge for edge, _ in path]
            current_car_summary += f'Car {agent_id} ({start_node}, {end_node}), arrived at t={arrival_time}, with path {formatted_path}'
            output_string += current_car_summary
        
        output_string += f'\nAverage congestion is {average_congestion}'
        output_string += f'\nTotal congestion is {total_congestion}'

        return output_string
    
    def calculate_total_congestion(self):
        """
        helper to calculate total confestion using detailed_path_record
        """
        result = 0

        for agent_id, path in self.detailed_path_record.items():
            for edge, congestion in path:
                result += congestion
        
        return result

class SmartBrainSimulator:
    """
    Smart Brain discrete-event simulator.
    Subclass and override handle(event_id, payload) with a switch-case.
    """
    def __init__(self, graph, start_time: float = 0.0) -> None:
        self.now = float(start_time)
        self._queue: list[Tuple[float, int, Any, Any, EventHandle]] = []
        self._seq = itertools.count()
        self._stopped = False
        self.events_processed = 0
        self.graph = graph
        # self.num_cars_on_edge_at_t = prepopulate_num_cars_at_t(graph)
        # self.initial_dijkstra_paths = {} # tracks our intially calcuated paths for each agents
        # self.detailed_path_record = {} # see paper for description
        # self.end_times = {} # agent_id: time it reached its destination node
        
        self.available_tasks = [] # format is list of 5-tuples (task-id, location (vertex-id), appear-time, target-time, reward)
        self.available_dashers = [] # list of 3-tuple dashers
        self.total_system_score = 0

    def schedule_at(self, time: float, event_id: Any, payload: Any = None) -> EventHandle:
        if time < self.now:
            raise ValueError("Cannot schedule in the past")
        seq = next(self._seq)
        h = EventHandle()
        heapq.heappush(self._queue, (float(time), seq, event_id, payload, h))
        return h

    def _pop_next(self):
        while self._queue:
            time, seq, event_id, payload, h = heapq.heappop(self._queue)
            if not h.cancelled:
                return time, event_id, payload
            # skipped cancelled
        return None

    def step(self) -> bool:
        if self._stopped:
            return False
        item = self._pop_next()
        if item is None:
            return False
        time, event_id, payload = item
        self.now = time
        # dispatch to user-defined handler
        self.handle(event_id, payload)
        self.events_processed += 1
        return True

    def run(self, until: Optional[float] = None, max_events: Optional[int] = None) -> None:
        self._stopped = False
        processed = 0
        while not self._stopped:
            if not self._queue:
                break
            if until is not None and self._queue[0][0] > until:
                break
            if max_events is not None and processed >= max_events:
                break
            if not self.step():
                break
            processed += 1

    def stop(self) -> None:
        self._stopped = True

    def handle(self, event_id: Any, payload: Any) -> None:
        """Override in a subclass with a simple switch (if/elif) on event_id."""
        current_time = self.now

        if event_id == 'T': # TaskArrival
            task_id = payload['task id']
            location = payload['location']
            appear_time = payload['appear time']
            target_time = payload['target time']
            reward = payload['reward']

            self.available_tasks.append((task_id, location, appear_time, target_time, reward))
            # print(f'{location=}')

        elif event_id == 'DA': # DasherArrival
            start_location = payload['start location']
            dasher_start_time = payload['start time']
            dasher_exit_time = payload['exit time']

            new_dasher_start_time = self.now + 2 # todo can change time maybe
            # print(f'{start_location=}')

            self.schedule_at(new_dasher_start_time, "DS", {'start location': start_location, 'start time': new_dasher_start_time, 'exit time': dasher_exit_time})
        
        elif event_id == 'DS': # DasherStart
            start_location = payload['start location']
            dasher_start_time = payload['start time']
            dasher_exit_time = payload['exit time']

            # print(f'DS: {start_location=}')

            self.available_dashers.append((start_location, dasher_start_time, dasher_exit_time))

            feasible_tasks = self.get_feasible_tasks_in_radius(start_location, 5, dasher_start_time, dasher_exit_time)
            # print(f'{feasible_tasks=}')

            # feasible task not found; make the dasher wait more time before searching for a task again
            if not feasible_tasks:
                if self.now < dasher_exit_time:
                    # only make dasher wait if they don't have to exit yet; otherwise, just let the dasher disappear
                    self.schedule_at(self.now, "DA", {'start location': start_location, 'start time': self.now, 'exit time': dasher_exit_time})
                return
            
            # print(f'{feasible_tasks=}')
            best_task = max(feasible_tasks, key=lambda task: task[2]) # highest reward or potentiall highest reward-to-time ratio task
            best_target_time = best_task[1]
            best_reward = best_task[2]
            best_ratio = best_reward / best_target_time
            best_index = feasible_tasks.index(best_task)

            # now, examine other tasks with rewards at least 70% of the best reward we've found so far
            # if any of those tasks have a better reward-to-time ratio, do the task with that highest ratio
            for i, potential_task in enumerate(filter(lambda task: task[2] >= 0.7*best_reward, feasible_tasks)):
                potential_task_location = potential_task[0] # index 0 is task's location
                potential_task_target_time = potential_task[1] # index 1 is task's target time
                potential_reward = potential_task[2]
                potential_ratio = potential_reward / potential_task_target_time
                if potential_ratio > best_ratio:
                    best_index = i
                    best_ratio = potential_ratio
                elif potential_ratio == best_ratio:
                    if potential_reward > best_reward:
                        best_index = i
                        best_ratio = potential_ratio
            
            # if not best_index == -1: # ie we DID find a task for this dasher

            try:
                # mark dasher as unavailable
                self.available_dashers.remove((start_location, dasher_start_time, dasher_exit_time))
            except Exception:
                print('dasher not available')

            new_start_location = feasible_tasks[best_index][0] # dasher's new location is old task's location
            new_dasher_start_time = feasible_tasks[best_index][1] # dasher's new start time is old task's target time
            dasher_exit_time = payload['exit time'] # dasher's exit time stays same

            actual_reward = feasible_tasks[best_index][2] # actual reward of the task the dasher is actually doing

            # Schedule the next DasherArrival event
            self.schedule_at(new_dasher_start_time, "DA", {'start location': new_start_location, 'start time': new_dasher_start_time, 'exit time': dasher_exit_time})

            # Add task's reward to the total-system-score
            self.total_system_score += feasible_tasks[best_index][2]

            # print(f'{self.available_tasks=}')
            # mark task as completed by removing it from available_tasks list
            self.available_tasks = [t for t in self.available_tasks if not (t[1] == new_start_location and t[3] == new_dasher_start_time and t[4] == actual_reward)]
                
            if best_index == -1:
                # feasible task not found; make the dasher wait more time before searching for a task again
                self.schedule_at(self.now, "DA", {'start location': new_start_location, 'start time': self.now, 'exit time': dasher_exit_time})
    
    def get_feasible_tasks_in_radius(self, start_node, radius, dasher_start_time, dasher_exit_time):
        """
        Given a node number and a radius, return list of nodes that are within that radius on a graph.

        :param start_node: - node id of where the center of the radius we want to check is
        :param radius: - numerical num of radius we want to check around the start_node
        :param dasher_start_time: - when dasher is starting to look for tasks. this param helps determine if a task will expire before the dasher can reach it, ie not be feasible
        :param dasher_exit_time: - when dasher must leave simulation. this param helps determine if dasher has to leave before a task is completed, ie that task is not feasible for this dasher

        :return: - a list of nodes with feasible (ie time-realistic) tasks in the format [(node, target time, reward), (node, target time, reward), etc.]
        """
        visited = set()
        queue = [(start_node, 0)] # format [(node, level), (node, level), etc.]
        result = [] # [(node, target time, reward), (node, target time, reward), etc.]

        while queue:
            current_node, current_level = queue.pop(0)

            if current_level < radius:
                for neighbor, _ in ((graph.getNeighbors(current_node)) + [(start_node, -1)]): # force start_node to be a "neighbor" so we can check if it has tasks
                    neighbor = str(neighbor)
                    # print(f'{neighbor=}')
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # print(f'{neighbor=}')
                        queue.append((neighbor, current_level + 1))

                        # check if there's any tasks at neighbor node (and possibly start node too)
                        potential_tasks_at_neighbor = list(filter(lambda task: str(task[1]) == str(neighbor), self.available_tasks))
                        # print(f'{potential_tasks_at_neighbor=}')
                        if potential_tasks_at_neighbor:
                            for potential_task in potential_tasks_at_neighbor:
                                task_target_time = potential_task[3]
                                task_appear_time = potential_task[2]
                                reward = potential_task[4]

                                # check if this potential task is feasible, ie can be completed in time
                                if (current_level + dasher_start_time <= task_target_time) \
                                    and (current_level + dasher_start_time <= dasher_exit_time) \
                                    and (task_appear_time <= dasher_start_time + 2) \
                                    and (dasher_exit_time >= task_target_time):
                                    # print("here")
                                    # btw, current_level = amount of time for dasher to reach "neighbor" from start_node
                                    result.append((neighbor, task_target_time, reward))
        # print(f'{result=}')
        return result


    def final_results_string(self):
        """
        Function to calculate final statistics and make correct format for output printing.
        Only invoke this function after the simulator is done running.
        """
        total_congestion = self.calculate_total_congestion()
        average_congestion = total_congestion / len(self.detailed_path_record)

        output_string = ""

        # now construct strings of this format: "Car 1 (start_node, end_node), arrived at t=.., with path (e1,t1), (e2,t2), ..."
        for agent_id in range(len(self.detailed_path_record)):
            if agent_id not in self.detailed_path_record:
                # if there's no possible path for this agent
                continue
            path = self.detailed_path_record[agent_id]
            current_car_summary = '\n'
            start_node = path[0][0][0]
            end_node = path[-1][0][1]
            arrival_time = self.end_times[agent_id]
            formatted_path = [edge for edge, _ in path]
            current_car_summary += f'Car {agent_id} ({start_node}, {end_node}), arrived at t={arrival_time}, with path {formatted_path}'
            output_string += current_car_summary
        
        output_string += f'\nAverage congestion is {average_congestion}'
        output_string += f'\nTotal congestion is {total_congestion}'

        return output_string
    
    def calculate_total_congestion(self):
        """
        helper to calculate total confestion using detailed_path_record
        """
        result = 0

        for agent_id, path in self.detailed_path_record.items():
            for edge, congestion in path:
                result += congestion
        
        return result


def baseline_dispatch_dashers(fname, base_sim: BaselineSimulator):
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for dasher_line in reader:
            start_location = dasher_line[0]
            start_time = int(dasher_line[1])
            exit_time = int(dasher_line[2])
            base_sim.schedule_at(start_time, "D", {'start location': str(start_location), 'start time': start_time, 'exit time': exit_time})


def smart_dispatch_dashers(fname, smart_sim: SmartBrainSimulator):
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for dasher_line in reader:
            start_location = dasher_line[0]
            start_time = int(dasher_line[1])
            exit_time = int(dasher_line[2])
            smart_sim.schedule_at(start_time, "DA", {'start location': str(start_location), 'start time': start_time, 'exit time': exit_time})


def schedule_tasks(fname, base_sim):
    task_payloads = []
    with open(fname, "r") as f:
        task_payloads = json.load(f)
    
    for payload in task_payloads:
        time = payload['target time']
        base_sim.schedule_at(time, "T", payload)



if __name__ == "__main__":
    ############## BASELINE SIMULATOR
    graph = read_graph("grid100.txt")
    base_sim = BaselineSimulator(graph)
    schedule_tasks("tasks_random_rewards_and_times.json", base_sim)
    baseline_dispatch_dashers("dashers.csv", base_sim)
    # base_sim.schedule_at(0, "T", {'task id': 1, 'location': 1, 'appear time': 0, 'target time': 1000, 'reward': 10})
    # base_sim.schedule_at(0, "T", {'task id': 2, 'location': 10, 'appear time': 0, 'target time': 1000, 'reward': 10})
    base_sim.run()
    # print(f'{base_sim.available_tasks=}')
    # print(f'{base_sim.available_dashers=}')
    print(f'{base_sim.total_system_score=}')

    # base_sim = BaselineSimulator(graph)
    # base_sim.schedule_at(0, "T", {'task id': 2, 'location': 5, 'appear time': 0, 'target time': 8, 'reward': 10})
    # base_sim.schedule_at(0, "T", {'task id': 1, 'location': 2, 'appear time': 0, 'target time': 3, 'reward': 10})
    # base_sim.schedule_at(0, "D", {'start location': 1, 'start time': 0, 'exit time': 10})
    # base_sim.schedule_at(0, "D", {'start location': 1, 'start time': 1, 'exit time': 9})
    # c

    # base_sim.run()
    # print('\nBaseline:')
    # print(base_sim.final_results_string())

    ########## SMART BRAIN SIMULATOR
    graph = read_graph("grid100.txt")
    smart_sim = SmartBrainSimulator(graph)
    # smart_sim.schedule_at(0, "T", {'task id': 1, 'location': '1', 'appear time': 0, 'target time': 10, 'reward': 100})
    # smart_sim.schedule_at(0, "T", {'task id': 2, 'location': '50', 'appear time': 0, 'target time': 10, 'reward': 10})
    # smart_sim.schedule_at(0, "T", {'task id': 3, 'location': '5', 'appear time': 2, 'target time': 10, 'reward': 101})
    # smart_sim.schedule_at(0, "DA", {'start location': '5', 'start time': 1, 'exit time': 50})
    # smart_sim.schedule_at(0, "DA", {'start location': '49', 'start time': 0, 'exit time': 50})
    # smart_sim.schedule_at(0, "DA", {'start location': '60', 'start time': 0, 'exit time': 50})



    #### tests emma added #####
    # smart_sim.schedule_at(0, "T", {'task id': 1, 'location': '0', 'appear time': 10, 'target time': 25, 'reward': 45})
    # smart_sim.schedule_at(0, "T", {'task id': 2, 'location': '32', 'appear time': 6, 'target time': 16, 'reward': 23})
    # smart_sim.schedule_at(0, "T", {'task id': 3, 'location': '9', 'appear time': 13, 'target time': 30, 'reward':  90})
    # smart_sim.schedule_at(0, "DA", {'start location': '20', 'start time': 5, 'exit time': 50})
    # smart_sim.schedule_at(0, "DA", {'start location': '37', 'start time': 0, 'exit time': 50})
    # smart_sim.schedule_at(0, "DA", {'start location': '6', 'start time': 4, 'exit time': 50})


    feas_tasks = smart_sim.get_feasible_tasks_in_radius('5', 5, 0, 20000)
    schedule_tasks("tasks_random_rewards_and_times.json", smart_sim)
    smart_dispatch_dashers("dashers.csv", smart_sim)
    smart_sim.run()
    print(f'{smart_sim.total_system_score=}')


 ### Machine Learning Forecasting ###


# # ============================
# # 1. Load and parse TXT file
# # ============================
#     rows = []

#     with open("available_tasks.txt", "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             # Each line looks like:
#             # (1080, 'T', {'task id': 1, 'location': '38', ...})
#             timestamp, tag, data = ast.literal_eval(line)

#             row = {
#                 "timestamp": int(timestamp),
#                 "task_id": int(data["task id"]),
#                 "location": int(data["location"]),
#                 "appear_time": int(data["appear time"]),
#                 "target_time": int(data["target time"]),
#                 "reward": int(data["reward"]),
#             }
#             rows.append(row)

#     df = pd.DataFrame(rows)
#     print("Loaded dataset:")
#     print(df.head())

#     # ============================
#     # 2. Features/target
#     # ============================
#     X = df[["task_id", "location", "appear_time", "target_time"]]
#     y = df["reward"]

#     # ============================
#     # 3. Train/test split
#     # ============================
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # ============================
#     # 4. Train XGBoost model
#     # ============================
#     model = XGBRegressor(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.1,
#     )

#     model.fit(X_train, y_train)

#     # ============================
#     # 5. Evaluate
#     # ============================
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)

#     print("\nRMSE:", rmse)
