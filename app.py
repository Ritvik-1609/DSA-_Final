from flask import Flask, request, jsonify, render_template
import networkx as nx
import osmnx as ox
import heapq
from geopy.distance import great_circle
import time

app = Flask(__name__)

# Global variable to store the graph
current_graph = None

def haversine_distance(node1, node2, G):
    """
    Calculate the Haversine distance between two nodes in a graph.

    Parameters
    ----------
    node1 : int
        The identifier for the first node.
    node2 : int
        The identifier for the second node.
    G : networkx.Graph
        The graph containing the nodes.

    Returns
    -------
    float
        The Haversine distance between the two nodes in meters.
    """
    coord1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
    coord2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
    return great_circle(coord1, coord2).meters

def astar_greedy(G, start, goal):
    """
    Perform the A* Greedy algorithm to find the shortest path in a graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph in which to find the path.
    start : int
        The identifier for the start node.
    goal : int
        The identifier for the goal node.

    Returns
    -------
    tuple
        A tuple containing the list of node identifiers for the shortest path and the time taken in milliseconds.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in G.nodes}
    f_score[start] = haversine_distance(start, goal, G)

    start_time = time.time()  # Start the timer

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            end_time = time.time()  # End the timer
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]
            time_taken = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds
            return path, time_taken

        for neighbor in G.neighbors(current):
            tentative_g_score = g_score[current] + G.edges[current, neighbor, 0]['length']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + haversine_distance(neighbor, goal, G)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    end_time = time.time()  # End the timer in case no path is found
    time_taken = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds
    return [], time_taken

def dijkstra(G, start, goal):
    """
    Perform Dijkstra's algorithm to find the shortest path in a graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph in which to find the path.
    start : int
        The identifier for the start node.
    goal : int
        The identifier for the goal node.

    Returns
    -------
    tuple
        A tuple containing the list of node identifiers for the shortest path and the time taken in milliseconds.
    """
    queue = [(0, start)]
    distances = {node: float('inf') for node in G.nodes}
    distances[start] = 0
    came_from = {}

    start_time = time.time()  # Start the timer

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            end_time = time.time()  # End the timer
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path = path[::-1]
            time_taken = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds
            return path, time_taken

        for neighbor in G.neighbors(current_node):
            weight = G.edges[current_node, neighbor, 0]['length']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                came_from[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    end_time = time.time()  # End the timer in case no path is found
    time_taken = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds
    return [], time_taken

@app.route('/')
def index():
    """
    Render the index page.

    Returns
    -------
    str
        The rendered HTML content of the index page.
    """
    return render_template('index.html')

@app.route('/set_city', methods=['POST'])
def set_city():
    """
    Set the current city and load its graph.

    Returns
    -------
    flask.Response
        A JSON response containing the bounding box of the city.
    """
    global current_graph
    data = request.json
    city_name = data['city']
    
    # Fetch the graph and bounding box for the given city
    G = ox.graph_from_place(city_name, network_type='drive')
    current_graph = G
    nodes = G.nodes(data=True)
    lats = [node['y'] for node_id, node in nodes]
    lngs = [node['x'] for node_id, node in nodes]
    bounds = {
        'southWest': [min(lats), min(lngs)],
        'northEast': [max(lats), max(lngs)]
    }
    return jsonify(bounds=bounds)

@app.route('/get_path', methods=['POST'])
def get_path():
    """
    Get the shortest path between two points using both A* and Dijkstra's algorithms.

    Returns
    -------
    flask.Response
        A JSON response containing the paths and times taken by both algorithms.
    """
    global current_graph
    if current_graph is None:
        return jsonify({'error': 'No city selected'}), 400
    
    data = request.json
    start_coords = data['start']
    end_coords = data['end']
    
    G = current_graph
    start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
    
    astar_path, astar_time = astar_greedy(G, start_node, end_node)
    dijkstra_path, dijkstra_time = dijkstra(G, start_node, end_node)
    
    astar_path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in astar_path]
    dijkstra_path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in dijkstra_path]

    return jsonify(
        astar_path=astar_path_coords, 
        astar_time=astar_time,
        dijkstra_path=dijkstra_path_coords, 
        dijkstra_time=dijkstra_time
    )

if __name__ == '__main__':
    app.run(debug=True)
