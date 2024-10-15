import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import heapq 
from memory_profiler import memory_usage

# load the adjacencies
def load_adjacency_list(filename):
    adj_list = nx.Graph()  
    try:
        with open(filename, 'r') as file:
            for line in file:
                city1, city2 = line.strip().split()
                adj_list.add_edge(city1, city2)  
    except FileNotFoundError:
        print(f"Error: Could not open file {filename}")
    return adj_list

# load city coordinates 
def load_city_coordinates(filename):
    coordinates = {}
    try:
        df = pd.read_csv(filename, header=None)
        for index, row in df.iterrows():
            city, lat, lon = row
            coordinates[city] = (float(lat), float(lon))  
    except FileNotFoundError:
        print(f"Error: Could not open file {filename}")
    return coordinates

# calculate distance
def calculate_distance(coord1, coord2):

    R = 6371  
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# Breadth-First Search
def bfs_with_visualization(graph, start, end, coordinates):
    def bfs_implementation(graph, start, end, coordinates):
        visited = []
        queue = [(start, [start])]
        start_time = time.time()
        total_distance = 0

        while queue:
            current_city, path = queue.pop(0)
            visited.append(current_city)
            visualize_graph(graph, visited, path, coordinates)

            if current_city == end:
                for i in range(len(path) - 1):
                    total_distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]])
                end_time = time.time()
                total_time = end_time - start_time
                return path, total_time, total_distance

            for neighbor in graph.neighbors(current_city):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None, None, None

    mem_usage, (path, total_time, total_distance) = memory_usage(
        (bfs_implementation, (graph, start, end, coordinates)),
        max_iterations=1,
        retval=True
    )
    max_mem = max(mem_usage) - min(mem_usage)
    return path, total_time, total_distance, max_mem

# Depth-First Search
def dfs_with_visualization(graph, start, end, coordinates):
    def dfs_implementation(graph, start, end, coordinates):
        visited = []
        stack = [(start, [start])]
        start_time = time.time()
        total_distance = 0

        while stack:
            current_city, path = stack.pop()
            visited.append(current_city)
            visualize_graph(graph, visited, path, coordinates)

            if current_city == end:
                for i in range(len(path) - 1):
                    total_distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]])
                end_time = time.time()
                total_time = end_time - start_time
                return path, total_time, total_distance

            for neighbor in graph.neighbors(current_city):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None, None, None

    mem_usage, (path, total_time, total_distance) = memory_usage(
        (dfs_implementation, (graph, start, end, coordinates)),
        max_iterations=1,
        retval=True
    )
    max_mem = max(mem_usage) - min(mem_usage)
    return path, total_time, total_distance, max_mem

# Iterative Deepening Depth-First Search
def id_dfs_with_visualization(graph, start, end, coordinates):
    def id_dfs_implementation(graph, start, end, coordinates):
        start_time = time.time()
        total_distance = 0
        depth_limit = 0

        while True:
            visited = []
            stack = [(start, [start], 0)]

            while stack:
                current_city, path, depth = stack.pop()
                if depth > depth_limit:
                    continue
                visited.append(current_city)
                visualize_graph(graph, visited, path, coordinates)

                if current_city == end:
                    for i in range(len(path) - 1):
                        total_distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]])
                    end_time = time.time()
                    total_time = end_time - start_time
                    return path, total_time, total_distance

                for neighbor in graph.neighbors(current_city):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], depth + 1))

            depth_limit += 1

        return None, None, None

    mem_usage, (path, total_time, total_distance) = memory_usage(
        (id_dfs_implementation, (graph, start, end, coordinates)),
        max_iterations=1,
        retval=True
    )
    max_mem = max(mem_usage) - min(mem_usage)
    return path, total_time, total_distance, max_mem

# Best-First Search
def best_first_search(graph, start, end, coordinates):
    def bfs_implementation(graph, start, end, coordinates):
        visited = set()
        priority_queue = []
        heapq.heappush(priority_queue, (0, start, [start]))
        start_time = time.time()
        total_distance = 0

        while priority_queue:
            _, current_city, path = heapq.heappop(priority_queue)
            if current_city in visited:
                continue
            visited.add(current_city)
            visualize_graph(graph, visited, path, coordinates)

            if current_city == end:
                for i in range(len(path) - 1):
                    total_distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]])
                end_time = time.time()
                total_time = end_time - start_time
                return path, total_time, total_distance

            for neighbor in graph.neighbors(current_city):
                if neighbor not in visited:
                    heuristic = calculate_distance(coordinates[neighbor], coordinates[end])
                    heapq.heappush(priority_queue, (heuristic, neighbor, path + [neighbor]))

        return None, None, None

    mem_usage, (path, total_time, total_distance) = memory_usage(
        (bfs_implementation, (graph, start, end, coordinates)),
        max_iterations=1,
        retval=True
    )
    max_mem = max(mem_usage) - min(mem_usage)
    return path, total_time, total_distance, max_mem  

# A* Search
def a_star_search(graph, start, end, coordinates):
    def astar_implementation(graph, start, end, coordinates):
        visited = set()
        priority_queue = []
        heapq.heappush(priority_queue, (0, start, [start], 0))
        start_time = time.time()
        total_distance = 0

        while priority_queue:
            f_cost, current_city, path, g_cost = heapq.heappop(priority_queue)
            if current_city in visited:
                continue
            visited.add(current_city)
            visualize_graph(graph, visited, path, coordinates)

            if current_city == end:
                total_distance = g_cost
                end_time = time.time()
                total_time = end_time - start_time
                return path, total_time, total_distance

            for neighbor in graph.neighbors(current_city):
                if neighbor not in visited:
                    new_g_cost = g_cost + calculate_distance(coordinates[current_city], coordinates[neighbor])
                    heuristic = calculate_distance(coordinates[neighbor], coordinates[end])
                    f_cost = new_g_cost + heuristic
                    heapq.heappush(priority_queue, (f_cost, neighbor, path + [neighbor], new_g_cost))

        return None, None, None

    mem_usage, (path, total_time, total_distance) = memory_usage(
        (astar_implementation, (graph, start, end, coordinates)),
        max_iterations=1,
        retval=True
    )
    max_mem = max(mem_usage) - min(mem_usage)
    return path, total_time, total_distance, max_mem 

# graph visualization
def visualize_graph(graph, visited, path, coordinates):
    plt.clf()  
    
    pos = {city: (coordinates[city][1], coordinates[city][0]) for city in graph.nodes()} 

    nx.draw(graph, pos, with_labels=True, node_size=500, font_size=10, font_weight='bold', edge_color='gray', node_color='lightblue')

    if visited:
        nx.draw_networkx_nodes(graph, pos, nodelist=visited, node_color='green')

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Search Visualization with Geographic Coordinates")
    plt.pause(0.75)




def main():
    adjacency_file = "D:\\SEMESTER 5\\COMP-SCI 461\\Search Program\\Adjacencies.txt"
    graph = load_adjacency_list(adjacency_file)

    coordinates_file = "D:\\SEMESTER 5\\COMP-SCI 461\\Search Program\\coordinates.csv"
    city_coordinates = load_city_coordinates(coordinates_file)

    print("Available cities:\n")
    available_cities = sorted(list(graph.nodes()))
    for idx, city in enumerate(available_cities, start=1):
        print(f"{idx}. {city}")

    print("\nEnter the number corresponding to the starting city:")
    start_city_index = int(input()) - 1
    start_city = available_cities[start_city_index]

    print("Enter the number corresponding to the destination city:")
    end_city_index = int(input()) - 1
    end_city = available_cities[end_city_index]

    while True:
        print("\nSelect the search method:")
        print("1. Breadth-First Search (BFS)")
        print("2. Depth-First Search (DFS)")
        print("3. Iterative Deepening Depth-First Search (ID-DFS)")
        print("4. Best-First Search (BestFS)")
        print("5. A* Search")
        method_choice = int(input("Enter your choice (1, 2, 3, 4, or 5): "))

        search_functions = {
            1: bfs_with_visualization,
            2: dfs_with_visualization,
            3: id_dfs_with_visualization,
            4: best_first_search,
            5: a_star_search
        }

        if method_choice in search_functions:
            path, total_time, total_distance, max_mem = search_functions[method_choice](graph, start_city, end_city, city_coordinates)
        else:
            print("Invalid choice. Please try again.")
            continue

        if path:
            print("\nFound Path: " + " -> ".join(path))
            print(f"Total time taken: {total_time:.2f} seconds")
            print(f"Total distance: {total_distance:.2f} kilometers")
            print(f"Memory usage: {max_mem:.2f} Mb")
        else:
            print("No route found.")

        retry = input("\nWould you like to try a different method? (y/n): ").strip().lower()
        if retry != 'y':
            break

plt.ion()
plt.figure(figsize=(12, 8))

if __name__ == "__main__":
    main()