import networkx as nx
from collections import deque

def load_twitter_graph(file_path):
    Gx = nx.DiGraph()  # Create a directed graph
    with open(file_path, 'r') as file:
        for line in file:
            a, b = map(int, line.strip().split())
            Gx.add_edge(a, b)  
    return Gx

def convert_to_undirected_reciprocal(Gx):
    Gx2 = nx.Graph()  # Create an empty undirected graph
    for a, b in Gx.edges():
        if Gx.has_edge(b, a):  
            Gx2.add_edge(a, b)  
    return Gx2

def bfs_crawl(Gx2, start_node, max_nodes):
    Gx2b = nx.Graph() 
    visited = set()  
    queue = deque([start_node])  
    visited.add(start_node)  
    Gx2b.add_node(start_node)  

    while queue:
        # Check if we have already reached the max node limit
        if len(Gx2b.nodes) >= max_nodes:
            break  

        current = queue.popleft()  

        # Get children of the current node and sort them as integers
        children = sorted(Gx2.neighbors(current), key=int)

        for child in children:
            if child not in visited:
                # Check if adding this child will exceed the limit
                if len(Gx2b.nodes) < max_nodes:
                    visited.add(child)  
                    queue.append(child)  
                    Gx2b.add_node(child)  
                    Gx2b.add_edges_from(Gx2.edges(child))  

    return Gx2b




def compute_graph_metrics(Gx2b, root_node, output_file):
    metrics = {}
    metrics['Root'] = root_node
    metrics['Number of Nodes'] = 10000
    metrics['Number of Edges'] = Gx2b.number_of_edges()
    
    # Ensure to only calculate diameter if the graph is connected
    if nx.is_connected(Gx2b):
        metrics['Diameter'] = nx.diameter(Gx2b)
    else:
        metrics['Diameter'] = float('inf')  # Or handle as you wish

    # Calculate average distance
    if metrics['Number of Nodes'] > 1:
        metrics['Average Distance'] = nx.average_shortest_path_length(Gx2b)
    else:
        metrics['Average Distance'] = 0

    metrics['Average Clustering Coefficient'] = nx.average_clustering(Gx2b)

    with open(output_file, 'a') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

# Main execution
if __name__ == "__main__":
    file_path = 'twitter_combined.txt' 
    output_file = 'social_network_analysis.txt'
    Gx = load_twitter_graph(file_path)  
    Gx2 = convert_to_undirected_reciprocal(Gx) 
    start_node = 56860418  
    Gx2b = bfs_crawl(Gx2, start_node, 10000)  

    # Compute metrics and output results
    metrics = compute_graph_metrics(Gx2b, start_node, output_file)
