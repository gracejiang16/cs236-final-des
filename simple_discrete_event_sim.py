from __future__ import annotations
import heapq
import itertools
import math
import random
from typing import Any, Optional, Tuple
from helpers import read_graph, prepopulate_num_cars_at_t

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

            feasible_task_index = -1
            smallest_time = float('inf')
            for i, potential_task in enumerate(self.available_tasks):
                task_location = potential_task[1] # index 1 is task's location
                _, projected_time = graph.dijkstra_shortest_path(int(start_location), task_location)

                task_target_time = potential_task[3] # index 3 is task's target time
                if (dasher_start_time + projected_time <= task_target_time)\
                    and (dasher_start_time + projected_time <= dasher_exit_time)\
                    and (projected_time < smallest_time):
                    
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
                base_sim.schedule_at(new_dasher_start_time, "D", {'start location': new_start_location, 'start time': new_dasher_start_time, 'exit time': dasher_exit_time})

                # Add task's reward to the total-system-score
                self.total_system_score += self.available_tasks[feasible_task_index][4]

                # mark task as completed by removing it from available_tasks list
                self.available_tasks.pop(feasible_task_index)
                
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

class AlgorithmXSimulator:
    """
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
        id = payload['id'] # AGENT id, not event id

        if event_id == 'A':
            destination = payload['end'] # get final destination node
            current_node = payload['arrival node']

            if current_node == destination:
                # the car has arrived at its destination, so it's done. record its end time
                self.end_times[id] = current_time
            else:
                # getting next edge to traverse by running Dijkstra from the current node
                path, _ = self.graph.dijkstra_shortest_path(current_node, destination)
                if not path:
                    # if path between these two nodes doesn't exist
                    return
                
                next_edge = (path[0], path[1])

                self.schedule_at(current_time, "D", {"edge": next_edge, "id": id, 'end': payload['end']})

        elif event_id == 'D':
            # first, update how many cars on the upcoming edge
            next_edge = payload['edge']
            self.num_cars_on_edge_at_t[(next_edge, current_time)] = self.num_cars_on_edge_at_t.get((next_edge, current_time), 0) + 1

            # then find how long it takes for this car to traverse this edge
            a, b = next_edge
            og_weight = self.graph.adj_list[a][b]
            new_congestion_time = self.num_cars_on_edge_at_t[(next_edge, current_time)] * og_weight

            arrival_time = current_time + new_congestion_time
            self.schedule_at(arrival_time, "A", {"id": id, "arrival node": b, "end": payload['end']})

            if id not in self.detailed_path_record:
                self.detailed_path_record[id] = [(next_edge, new_congestion_time)]
            else:
                self.detailed_path_record[id].append((next_edge, new_congestion_time))

        elif event_id == 'E':
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
                # this agent did not successfully find/traverse a path
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

def run_simulation_n_times(n, base=True):
    """
    Runs a simulation n number of times, given the simulator type boolean flag (which could be Baseline or AlgorithmX)
    
    :return: - tuple in format (totalcongestion, averagecongestion)
    """
    # graph = read_graph("testing/graph2.txt")
    graph = read_graph("input/graph1.txt")
    # agent_file = open("testing/agents2.txt", "r")
    agent_file = open("input/agents16.txt", "r")

    if base:
        for _ in range(n):
            base_sim = BaselineSimulator(graph)
            # read the agents
            for agent_id, agent_line in enumerate(agent_file):
                start_and_end_nodes = agent_line.strip().split(",")
                start_node = start_and_end_nodes[0]
                end_node = start_and_end_nodes[1]

                random_time = float(random.randint(1, 10)) # randomize initial arrivals of all agents/cars

                # run normal Dijkstra
                initial_path, initial_path_total_cost = graph.dijkstra_shortest_path(start_node, end_node)
                base_sim.initial_dijkstra_paths[agent_id] = initial_path # initial path is the predetermined list of nodes of path from start to end node

                if len(initial_path) <= 1:
                    # if a car is starting and ending at same node, ignore it cuz it doesn't affect congestion
                    # this car stays in one spot, so its total time is zero
                    base_sim.detailed_path_record[agent_id] = [((initial_path[0], initial_path[0]), 0)]
                if math.isinf(initial_path_total_cost):
                    continue # there is no path for these start and end nodes

                first_edge_in_path = (initial_path[0], initial_path[0]) # FIRST EDGE is initial arrival, so we "arrive" at the VERY first node, hence the "artificial" edge from the first node to itself.
                base_sim.schedule_at(random_time, "A", {"edge": first_edge_in_path, "id": agent_id})
            # agent_file.close()

            base_sim.run()
            total_congestion = base_sim.calculate_total_congestion()
            average_congestion = total_congestion / len(base_sim.detailed_path_record)
            print(f'{average_congestion=}; {total_congestion=}')
    else:
        for _ in range(n):
            algo_x_sim = AlgorithmXSimulator(graph)
            # read the agents
            for agent_id, agent_line in enumerate(agent_file):
                start_and_end_nodes = agent_line.strip().split(",")
                start_node = start_and_end_nodes[0]
                end_node = start_and_end_nodes[1]

                random_time = float(random.randint(1, 10)) # randomize initial arrivals of all agents/cars

                first_edge_in_path = (start_node, start_node) # FIRST EDGE is initial arrival, so we "arrive" at the VERY first node, hence the "artificial" edge from the first node to itself.
                algo_x_sim.schedule_at(random_time, "A", {"edge": first_edge_in_path, "id": agent_id, 'arrival node': start_node, 'end': end_node})

            algo_x_sim.run()
            total_congestion = algo_x_sim.calculate_total_congestion()
            average_congestion = total_congestion / len(algo_x_sim.detailed_path_record)
            print(f'{average_congestion=}; {total_congestion=}')
        agent_file.close()


if __name__ == "__main__":
    ############## BASELINE SIMULATOR
    graph = read_graph("/Users/gracejiang/Documents/Wellesley/4th Year/CS236/final/cs236-final-des/testing/graph2.txt")
    # graph = read_graph("input/grid100.txt")
    agent_file = open("/Users/gracejiang/Documents/Wellesley/4th Year/CS236/final/cs236-final-des/testing/agents2.txt", "r")
    # agent_file = open("input/agents100.txt", "r")

    base_sim = BaselineSimulator(graph)
    base_sim.schedule_at(0, "T", {'task id': 1, 'location': 1, 'appear time': 0, 'target time': 3, 'reward': 10})
    base_sim.schedule_at(0, "T", {'task id': 2, 'location': 5, 'appear time': 0, 'target time': 3, 'reward': 10})
    base_sim.schedule_at(0, "D", {'start location': 2, 'start time': 0, 'exit time': 10})
    base_sim.run()
    print(f'{base_sim.available_tasks=}')
    # read the agents
    # for agent_id, agent_line in enumerate(agent_file):
    #     start_and_end_nodes = agent_line.strip().split(",")
    #     start_node = start_and_end_nodes[0]
    #     end_node = start_and_end_nodes[1]

    #     random_time = float(random.randint(1, 10)) # randomize initial arrivals of all agents/cars

    #     # run normal Dijkstra
    #     initial_path, initial_path_total_cost = graph.dijkstra_shortest_path(start_node, end_node)
    #     base_sim.initial_dijkstra_paths[agent_id] = initial_path # initial path is the predetermined list of nodes of path from start to end node

    #     if len(initial_path) <= 1:
    #         # if a car is starting and ending at same node, ignore it cuz it doesn't affect congestion
    #         # this car stays in one spot, so its total time is zero
    #         if not initial_path:
    #             continue # there is no path from current start to end node: disregard
    #         base_sim.detailed_path_record[agent_id] = [((initial_path[0], initial_path[0]), 0)]
    #     if math.isinf(initial_path_total_cost):
    #         continue # there is no path for these start and end nodes

    #     first_edge_in_path = (initial_path[0], initial_path[0]) # FIRST EDGE is initial arrival, so we "arrive" at the VERY first node, hence the "artificial" edge from the first node to itself.
    #     base_sim.schedule_at(random_time, "A", {"edge": first_edge_in_path, "id": agent_id})
    # agent_file.close()

    # base_sim.run()
    # print('\nBaseline:')
    # print(base_sim.final_results_string())

    ########## ALGO X SIMULATOR
    # # graph = read_graph("testing/graph2.txt")
    # graph = read_graph("input/grid100.txt")
    # # agent_file = open("testing/agents2.txt", "r")
    # agent_file = open("input/agents100.txt", "r")

    # algo_x_sim = AlgorithmXSimulator(graph)
    # # read the agents
    # for agent_id, agent_line in enumerate(agent_file):
    #     start_and_end_nodes = agent_line.strip().split(",")
    #     start_node = start_and_end_nodes[0]
    #     end_node = start_and_end_nodes[1]

    #     random_time = float(random.randint(1, 10)) # randomize initial arrivals of all agents/cars

    #     first_edge_in_path = (start_node, start_node) # FIRST EDGE is initial arrival, so we "arrive" at the VERY first node, hence the "artificial" edge from the first node to itself.
    #     algo_x_sim.schedule_at(random_time, "A", {"edge": first_edge_in_path, "id": agent_id, 'arrival node': start_node, 'end': end_node})
    # agent_file.close()

    # algo_x_sim.run()
    # print('\nAlgorithmX:')
    # print(algo_x_sim.final_results_string())