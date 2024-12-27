import networkx as nx

# Function to create an undirected graph for Facebook
def create_facebook_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            a, b = map(int, line.strip().split())
            G.add_edge(a, b)
    return G

# Function to create a directed graph for Twitter
def create_twitter_graph(file_path):
    G = nx.DiGraph() 
    with open(file_path, 'r') as f:
        for line in f:
            a, b = map(int, line.strip().split())
            G.add_edge(a, b)
    return G

# Function to analyze the Facebook graph
def analyze_facebook_graph(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    diameter = nx.diameter(G) if nx.is_connected(G) else "Graph is not connected"
    average_distance = nx.average_shortest_path_length(G) if nx.is_connected(G) else "Graph is not connected"
    avg_clustering_coefficient = nx.average_clustering(G)
    
    # Calculate degrees and get top 10 users
    degrees = dict(G.degree())
    top_users = sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:10]
    
    return (num_nodes, num_edges, diameter, average_distance, avg_clustering_coefficient, top_users)

# Function to analyze the Twitter graph
def analyze_twitter_graph(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calculate in-degrees (followers)
    in_degrees = dict(G.in_degree())
    top_followers = sorted(in_degrees.items(), key=lambda item: item[1], reverse=True)[:10]
    
    return num_nodes, num_edges, top_followers

# Function to write results to a file
def write_results_to_file(facebook_results, twitter_results, output_file):
    with open(output_file, 'w') as f:
        f.write("Facebook Network Analysis:\n")
        f.write(f"Number of Nodes = {facebook_results[0]}\n")
        f.write(f"Number of Edges = {facebook_results[1]}\n")
        f.write(f"Diameter = {facebook_results[2]}\n")
        f.write(f"Average Distance = {facebook_results[3]}\n")
        f.write(f"Average Clustering Coefficient = {facebook_results[4]}\n")
        
        f.write("\nTop 10 Facebook Users:\n")
        for user, friends in facebook_results[5]:
            f.write(f"Facebook user {user} who has {friends} friends\n")
        
        f.write("\nTwitter Network Analysis:\n")
        f.write(f"Number of Nodes = {twitter_results[0]}\n")
        f.write(f"Number of Edges = {twitter_results[1]}\n")
        
        f.write("\nTop 10 Twitter Users:\n")
        for user, followers in twitter_results[2]:
            f.write(f"Twitter user {user} who has {followers} followers\n")

# Main function to execute the steps
if __name__ == "__main__":
    facebook_file_path = 'facebook_combined.txt' 
    twitter_file_path = 'twitter_combined.txt'  
    output_file = 'social_network_analysis.txt' 
    
    # Create the Facebook graph from the file
    Gf = create_facebook_graph(facebook_file_path)
    
    # Create the Twitter graph from the file
    Gx = create_twitter_graph(twitter_file_path)
    
    # Analyze both graphs
    facebook_results = analyze_facebook_graph(Gf)
    twitter_results = analyze_twitter_graph(Gx)
    
    # Write results to the output file
    write_results_to_file(facebook_results, twitter_results, output_file)
    
    print("Finish")
