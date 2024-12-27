import networkx as nx

# Step 1: Load the Twitter social network from the file
def load_twitter_graph(file_path):
    Gx = nx.DiGraph() 
    with open(file_path, 'r') as file:
        for line in file:
            a, b = map(int, line.strip().split())
            Gx.add_edge(a, b) 
    return Gx

# Step 2: Convert to undirected graph
def convert_to_undirected(Gx):
    Gx1 = nx.Graph()  
    for a, b in Gx.edges():
        Gx1.add_edge(a, b)  
    return Gx1

# Step 3: Calculate and print number of nodes and edges
def analyze_undirected_graph(Gx1, output_file):
    num_nodes = Gx1.number_of_nodes()
    num_edges = Gx1.number_of_edges()
    
    with open(output_file, 'a') as f:  
        f.write("Undirected Twitter Graph Analysis:\n")
        f.write(f"Number of Nodes: {num_nodes}\n")
        f.write(f"Number of Edges: {num_edges}\n\n")

# Main execution
if __name__ == "__main__":
    twitter_file = 'twitter_combined.txt' 
    output_file = 'social_network_analysis.txt'

    # Load the Twitter graph
    Gx = load_twitter_graph(twitter_file)

    # Convert to undirected graph
    Gx1 = convert_to_undirected(Gx)

    # Analyze the undirected graph
    analyze_undirected_graph(Gx1, output_file)

    print("Undirected Twitter graph analysis has been written to social_network_analysis.txt")
