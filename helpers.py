import csv
import random


class HeapNode():
    """
    Node in heap with value AND priority attributes.
    """
    def __init__(self, val, pri):
        self.value = val
        self.priority = pri  # lower numer means higher priority

class PriorityQueue:
    def __init__(self):
        self.queue = []  # internally represented by an array (min-heap) of HeapNodes
        
    def insert(self, item, priority):
        """
        inserts a new HeapNode with specified priority at the end of the heap, then heapify-ups it
        
        :param item: - actual node value
        :param priority: - node priority/key
        """
        node = HeapNode(item, priority)

        (self.queue).append(node)
        i = len(self.queue) - 1
        self.heapify_up(i)

    def heapify_up(self, i):
        """
        helper for insert method

        :param i: - (zero-based) index of current item to heapify up
        """
        if i > 0:
            parent_i = (i-1) // 2
            
            parent_key = self.queue[parent_i].priority
            child_key = self.queue[i].priority

            if parent_key > child_key:
                
                # swap parent and child HeapNodes
                temp = self.queue[parent_i]
                self.queue[parent_i] = self.queue[i]
                self.queue[i] = temp

                # then heapify up again
                self.heapify_up(parent_i)

    def heapify_down(self, i):
        """
        helper for extractMin method

        :param i: - (zero-based) index of current item to heapify down
        """

        length = len(self.queue)
        if 2 * i >= length:
            return
        
        elif 2 * i < length:
            left_i = (2 * i) + 1
            right_i = (2 * i) + 2

            if left_i >= length: # or right_i >= length:
                return # current node is a leaf, so we're done

            smallest_i = -1  # (temp) index of smallest key among left and right children
            if right_i >= length: # if there is no right child
                smallest_i = left_i
            else: # if there is a right child, must find if left or right is smaller
                if self.queue[left_i].priority < self.queue[right_i].priority:
                    smallest_i = left_i
                else:
                    smallest_i = right_i
            
        if self.queue[smallest_i].priority < self.queue[i].priority:
            # swap the current with its smallest key child
            temp = self.queue[smallest_i]
            self.queue[smallest_i] = self.queue[i]
            self.queue[i] = temp

            # then heapify down again
            self.heapify_down(smallest_i)

    def extractMin(self):
        """
        returns the top of the heap--the node that has the smallest key in the p-queue
        """
        if self.isEmpty():
            return None
        
        result = self.queue.pop(0)

        if len(self.queue) > 0:
            self.queue.insert(0, (self.queue.pop(-1))) # remove the new last item, put it at top of minheap
            self.heapify_down(0)
        return result

    def decreasePriority(self, item, priority):
        """
        To clarify, this is decreasing the Priority NUMBER, which means it has a higher priority (in English), so we should "heapify up"

        :param item: - node value of the node we want to decrease key for
        :param priority: - new lower key value to assign
        """
        # first, find index of HeapNode with value=item
        current_i = -1
        for i, heap_node in enumerate(self.queue):
            val = heap_node.value
            # print(f'{val=}, {item=}')
            if val == item:
                current_i = i
        
        if current_i == -1:
            raise Exception(f"Node {item} doesn't exist")
        
        self.queue[current_i].priority = priority
        self.heapify_up(current_i)

    def isEmpty(self):
        return len(self.queue) == 0
    
    def __str__(self):
        """
        assist in debugging
        """
        result = '['

        for node in self.queue:
            result = result + '(' + str(node.value) + ', ' + str(node.priority) + '), '

        return result + ']'
    
    def get_key_given_value(self, value):
        """
        helper function. given a node value, returns its priority/key
        """
        for node in self.queue:
            if node.value == value:
                return node.priority
        raise Exception(f'invalid value: {value}')

class WeightedGraph:
    """
    Weighted directed graph. A python Set holds all vertices, and a dict holds all edges with their weights, like an adjacency list
    """
    
    def __init__(self, vertices, edges):
        """
        :param vertices: - list of nodes (string)
        :param edges: - list of 3-tuples in the format (v1, v2, cost)

        internal representation is a SET of verticies and DICT of edges/weights as an adj list
        """
        self.vertices = set(vertices)
        self.adj_list = self.populate_adj_list(vertices, edges)

    def populate_adj_list(self, vertices, edges):
        """
        Returns a dict with format {node1 : {node2: weight2, node3: weight3} ...}
        to represent an adjacency list
        """
        result = {}

        for v in vertices:
            result[v] = {}
        
        for (a, b, weight) in edges:
            if a in result:
                result[a][b] = weight
            else:
                result[a] = {b: weight}
        
        return result

    def addNode(self, node):
        """
        Adds a node to graph. Raises exception if node is already in graph
        """
        if node in self.vertices:
            raise Exception("Node already exists")
        self.vertices.add(node)

    def addEdge(self, node1, node2, weight):
        """
        Adds edge to graph. Raises exception if edge contains nonexistent nodes
        """
        if not (node1 in self.vertices and node2 in self.vertices):
            raise Exception("Can't form edge between nonexistent nodes")
        
        if node1 in self.adj_list:
            self.adj_list[node1][node2] = weight
        else:
            self.adj_list[node1] = {node2: weight}

    def modifyWeight(self, node1, node2, weight):
        """
        Modifies weight of edge between node1 and node2.
        Raises exception if nodes don't exist or edge doesn't exist.
        """
        if not (node1 in self.vertices and node2 in self.vertices):
            raise Exception("One or both nodes do not exist")
        
        if node2 not in self.adj_list[node1]:
            raise Exception("Edge does not exist")
        
        self.adj_list[node1][node2] = weight

    def getNeighbors(self, node):
        """
        Returns list of neighboring nodes to param node.
        Raises exception if node doesn't exist
        """
        if node not in self.vertices:
            raise Exception("Node does not exist")
    
        return [(b, weight) for (b, weight) in (self.adj_list[node]).items()]
    
    def getNodes(self):
        """
        Returns list of all verticies in the graph
        """
        return list(self.vertices)

    def dijkstra_shortest_path(self, start_node, num_cars_on_edges=None):
        """
        :param start_node: - value of HeapNode to start at
        :param end_node: - value of HeapNode to at
        :param num_cars_on_edges: - dict telling you how many cars are using an edge. helpful for calculating cost of an edge
        """
        distances_so_far = self.prepopulate_distances()
        # print(f'{distances_so_far.__str__()=}')
        distances_so_far.decreasePriority(start_node, 0) # make distance from start to start zero

        predecessors = {}

        visited = set() # set of explored nodes

        final_distances = {} # to store shortest paths lengths we know for sure
    
        while not distances_so_far.isEmpty(): # while we still have nodes to explore
            current_shortest_node = distances_so_far.extractMin()
            current_shortest_node_value = current_shortest_node.value
            current_shortest_node_length = current_shortest_node.priority

            visited.add(current_shortest_node_value) # we've found shortest path for this node. it is done with the algo
            final_distances[current_shortest_node_value] = current_shortest_node_length # record length of shortest path to this current node

            # see if can find shorter path to current node's neighbors
            for (neighbor, weight) in self.getNeighbors(current_shortest_node_value):
                if neighbor in visited:
                    # if we've already found shortest path to neighbor, no need to check it again
                    continue
                potential_new_path_length = final_distances[current_shortest_node_value] + self.get_edge_cost(weight, (current_shortest_node_value, neighbor), num_cars_on_edges)
                old_path_length = distances_so_far.get_key_given_value(neighbor)
                if potential_new_path_length < old_path_length:
                    # if new path dist is smaller than what we already have, update
                    distances_so_far.decreasePriority(neighbor, potential_new_path_length)
                    predecessors[neighbor] = current_shortest_node_value

        return (final_distances)

    def get_edge_cost(self, weight, edge, num_cars_on_edge):
        """
        helper. given num of cars on an edge, and a weight, calculate cost of edge.
        cost of edge = num cars * weight; unless num cars=0, then cost=weight
        """
        if num_cars_on_edge is None:
            return weight # just normal Dijkstra
        
        num_cars = num_cars_on_edge[edge]
        if num_cars == 0:
            return weight
        return num_cars * weight

    def prepopulate_distances(self):
        """
        Intialize the priority queue so that all starting path lengths are infinity
        """
        result = PriorityQueue()
        all_vertices = self.getNodes()
        for v in all_vertices:
            result.insert(v, float('inf'))

        return result

    def extract_path(self, start_node, end_node, predecessors):
        """
        Given a start node and end node find the path through the predecessors heirarchy dictionary  
        """
        result = [end_node]
        cursor = end_node      # track which node in the path we are at
        
        while cursor != start_node:
            cursor = predecessors.get(cursor, None)
            if not cursor:
                # if a predecessor doesn't exist, return an empty list []
                return []
            result.insert(0, cursor)
        
        return result

#####################

def read_graph(fname):
    # Open the file
    file = open(fname, "r")
    # Read the first line that contains the number of vertices
    # numVertices is the number of vertices in the graph (n)
    numVertices = file.readline()


    vertices = set()
    edges = []
    # Next, read the edges and build the graph
    for line in file:
        # edge is a list of 3 indices representing a pair of adjacent vertices and the weight
        # edge[0] contains the first vertex (index between 0 and numVertices-1)
        # edge[1] contains the second vertex (index between 0 and numVertices-1)
        # edge[2] contains the weight of the edge (a positive integer)
        edge = line.strip().split(",")
        vertex_a = edge[0]
        vertex_b = edge[1]
        vertices.add(vertex_a)
        vertices.add(vertex_b)
        cost = int(edge[2])

        edges.append((vertex_a, vertex_b, cost))

    # Use the edge information to populate your graph object
    graph = WeightedGraph(vertices, edges)
    
    # Close the file safely after done reading
    file.close()
    return graph


def prepopulate_num_cars_at_t(graph):
    """
    helper function to set initial num of cars on each edge at *timestamp 0* in graph to zero
    """
    result = {}
    adj_list = graph.adj_list
    for a, neighbors in adj_list.items():
        for neighbor in neighbors:
            result[((a, neighbor), 0)] = 0
    return result

def dispatch_dashers(fname, base_sim):
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for dasher_line in reader:
            start_location = dasher_line[0]
            start_time = int(dasher_line[1])
            exit_time = int(dasher_line[2])
            base_sim.schedule_at(start_time, "D", {'start location': str(start_location), 'start time': start_time, 'exit time': exit_time})


# def schedule_tasks(fname, base_sim):
#     with open(fname, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)
#         task_id = 1
#         for dasher_line in reader:
#             user_id, vertex, _, time = dasher_line
#             user_id, vertex, time = int(user_id), int(vertex), int(time)
#             appear_time = time - (random.randint(5, 30))
#             reward = random.randint(1, 100) 
#             base_sim.schedule_at(time, "T", {'task id': task_id, 'location': vertex, 'appear time': appear_time, 'target time': time, 'reward': reward})
#             task_id += 1
