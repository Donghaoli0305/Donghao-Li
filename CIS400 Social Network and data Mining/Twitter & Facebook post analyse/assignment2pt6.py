import networkx as nx

# Step 1: Load the Twitter social network from the file (same as before)
def load_twitter_graph(file_path):
    Gx = nx.DiGraph() 
    with open(file_path, 'r') as file:
        for line in file:
            a, b = map(int, line.strip().split())
            Gx.add_edge(a, b) 
    return Gx

# Step 2: Convert to undirected graph based on reciprocal connections
def convert_to_undirected_reciprocal(Gx):
    Gx2 = nx.Graph() 
    for a, b in Gx.edges():
        if Gx.has_edge(b, a):  
            Gx2.add_edge(a, b)  
    return Gx2

# Step 3: Calculate and print number of nodes and edges
def analyze_undirected_graph_reciprocal(Gx2, output_file):
    num_nodes = Gx2.number_of_nodes()
    num_edges = Gx2.number_of_edges()
    
    with open(output_file, 'a') as f: 
        f.write("Undirected Twitter Graph (Reciprocal) Analysis:\n")
        f.write(f"Number of Nodes: {num_nodes}\n")
        f.write(f"Number of Edges: {num_edges}\n\n")

# Main execution
if __name__ == "__main__":
    twitter_file = 'twitter_combined.txt'
    output_file = 'social_network_analysis.txt'

    # Load the Twitter graph
    Gx = load_twitter_graph(twitter_file)

    # Convert to undirected graph based on reciprocal friends
    Gx2 = convert_to_undirected_reciprocal(Gx)

    # Analyze the reciprocal undirected graph
    analyze_undirected_graph_reciprocal(Gx2, output_file)

    print("Reciprocal undirected Twitter graph analysis has been written to social_network_analysis.txt")
