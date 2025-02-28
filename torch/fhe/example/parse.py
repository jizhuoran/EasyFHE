import re
import networkx as nx

class MetaInfo:
    def __init__(self, cur_limbs, noise_deg):
        self.cur_limbs = cur_limbs
        self.noise_deg = noise_deg

    def __repr__(self):
        return f"(cur_limbs={self.cur_limbs}, noise_deg={self.noise_deg})"


def parse_code_to_graph(code):
    # Create an empty directed graph
    graph = nx.DiGraph()
    
    # Regular expression to match the pattern of the assignments
    assignment_pattern = re.compile(r'(\w+) = (.*)')
    
    # Regular expression to detect function calls (operation nodes)
    func_pattern = re.compile(r'(\w+)\(([^)]+)\)')
    
    # Define a dictionary to store the nodes and their inputs
    node_info = {}
    
    for line in code.splitlines():
        line = line.strip()
        
        # Ignore comments (lines starting with '#')
        if line.startswith("#") or not line:
            continue
        
        # Remove inline comments
        line = line.split('#')[0].strip()

        # Parse the assignment to extract the node and its assigned value
        match = assignment_pattern.match(line)
        if match:
            node_name = match.group(1)
            expression = match.group(2)
            
            # Check if it's a function call (operation)
            func_match = func_pattern.search(expression)
            if func_match:
                operation = func_match.group(1)
                inputs = [input.strip() for input in func_match.group(2).split(',')]
                # Add the operation to the graph with the node as the output
                graph.add_node(node_name, operation=operation, inputs=inputs)
                
                # Add edges between nodes (input -> output)
                for input_node in inputs:
                    if 'NODE' in input_node:
                        graph.add_edge(input_node, node_name)
            else:
                # If it's not an operation, treat it as a simple assignment
                graph.add_node(node_name, operation="assignment", inputs=[expression])
    
    # Return the constructed graph
    return graph


def calculate_metadata(operation, inputs, metadata_dict):
    if operation in['assignment', 'homo_rotate']:
        CT0 = metadata_dict[inputs[0]]
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg)
    elif operation == 'homo_add':
        CT0, CT1 = metadata_dict[inputs[0]], metadata_dict[inputs[1]]
        return MetaInfo(min(CT0.cur_limbs, CT1.cur_limbs), max(CT0.noise_deg, CT1.noise_deg))
    elif operation == 'homo_mul_scalar_double':
        CT0 = metadata_dict[inputs[0]]
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg + 1)
    elif operation == 'homo_rescale':
        CT0, scale_level = metadata_dict[inputs[0]], int(inputs[1])
        return MetaInfo(CT0.cur_limbs - scale_level, CT0.noise_deg - scale_level)
    elif operation == 'mod_raise':
        CT0, raise_level = metadata_dict[inputs[0]], int(inputs[1])
        return MetaInfo(raise_level, CT0.noise_deg)
    else:
        return MetaInfo(-1, -1)
    

def assign_metadata(graph, start_node, end_node):
    # Dictionary to store metadata for each node
    metadata_dict = {"IN_NODE" : MetaInfo(2, 1)}

    # Perform a breadth-first search (BFS) to traverse the graph from start_node to end_node
    queue = [start_node]
    visited = set()

    while queue:
        current_node = queue.pop(0)

        if current_node == end_node:
            break  # Stop if we reach NODE_OUT

        # Skip nodes we've already processed
        if current_node in visited:
            continue
        visited.add(current_node)

        # Retrieve the operation and inputs for the current node
        operation = graph.nodes[current_node].get("operation")
        inputs = graph.nodes[current_node].get("inputs")
        
        print(current_node)
        calculate_metadata(operation, inputs, metadata_dict)
        metadata_dict[current_node] = calculate_metadata(operation, inputs, metadata_dict)

        # Add successors (dependent nodes) to the queue for BFS traversal
        print("  Successors: ", end="")
        for successor in graph.successors(current_node):
            print(successor, end=" ")
            queue.append(successor)
        print()

    return metadata_dict


def print_graph_info(graph):
    for node in graph.nodes:
        print(f"Node: {node}")
        print(f"  Operation: {graph.nodes[node].get('operation')}")
        print(f"  Inputs: {graph.nodes[node].get('inputs')}")
        print(f"  Outputs: {[n for n in graph.successors(node)]}")
        print()

# Sample code string
with open("sample_code.txt", "r") as f:
    code = f.read()

# Parse the code and construct the graph
graph = parse_code_to_graph(code)

# Print the information about the graph
print_graph_info(graph)

# Define the start and end nodes
start_node = "NODE57"  # NODE_IN
end_node = "NODE102"   # NODE_OUT

# Calculate metadata for each node along the path from NODE_IN to NODE_OUT
metadata = assign_metadata(graph, start_node, end_node)

# Print the metadata for each node in the path
for node, data in metadata.items():
    print(f"Node: {node}")
    print(f"  data: {data}")
    print()
