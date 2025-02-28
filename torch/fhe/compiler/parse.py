import re
import networkx as nx

class MetaInfo:
    def __init__(self, cur_limbs, noise_deg):
        self.cur_limbs = cur_limbs
        self.noise_deg = noise_deg

    def __repr__(self):
        return f"(cur_limbs={self.cur_limbs}, noise_deg={self.noise_deg})"


def parse_code_to_graph(code):
    """
    Parse the provided code and construct a graph of operations.
    This handles multiple outputs from the same operation (e.g., NODE230, NODE231).
    """
    graph = nx.DiGraph()
    
    # Regular expression to match the pattern of the assignments
    assignment_pattern = re.compile(r'(\w+)(?:, (\w+))? = (.*)')  # Allows for multiple outputs
    
    # Regular expression to detect function calls (operation nodes)
    func_pattern = re.compile(r'(\w+)\(([^)]+)\)')
    
    for line in code.splitlines():
        line = line.strip()
        
        if not line:
            continue
        
        # Parse the assignment to extract the nodes and the assigned value
        match = assignment_pattern.match(line)
        if match:
            node_name1 = match.group(1)
            node_name2 = match.group(2) if match.group(2) else None
            expression = match.group(3)
            
            # Check if it's a function call (operation)
            func_match = func_pattern.search(expression)
            if func_match:
                operation = func_match.group(1)
                inputs = [input.strip() for input in func_match.group(2).split(',')]
                
                # Add nodes and their relationships to the graph
                graph.add_node(node_name1, operation=operation, inputs=inputs)
                if node_name2:
                    graph.add_node(node_name2, operation=operation, inputs=inputs)
                
                # Add edges between input nodes and the output nodes
                for input_node in inputs:
                    if 'NODE' in input_node:
                        graph.add_edge(input_node, node_name1)
                        if node_name2:
                            graph.add_edge(input_node, node_name2)
            else:
                # For assignments with no operation
                graph.add_node(node_name1, operation="assignment", inputs=[expression])
                if node_name2:
                    graph.add_node(node_name2, operation="assignment", inputs=[expression])
    
    return graph


def calculate_metadata(graph, metadata, node):
    operation = graph.nodes[node].get("operation")
    inputs = graph.nodes[node].get("inputs")

    if operation in['assignment', 'homo_rotate', 'extract_cv', 'key_switch_P_ext', 'modup_to_ext', 'eval_fast_rotate', 'moddown_from_ext', '_cipher_automorphism', 'homo_add_scalar_double', 'homo_mul_scalar_int']:
        CT0 = metadata[inputs[0]]
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg)
    elif operation in ['homo_add', 'homo_sub']:
        CT0, CT1 = metadata[inputs[0]], metadata[inputs[1]] #this is wrong
        if metadata["RESCALE_TECH"] == "FLEXIBLEAUTO":
            if CT0.cur_limbs > CT1.cur_limbs:
                target_libms = CT1.cur_limbs
                target_noise_deg = CT1.noise_deg
            elif CT0.cur_limbs < CT1.cur_limbs:
                target_libms = CT0.cur_limbs
                target_noise_deg = CT0.noise_deg
            else:
                target_libms = CT0.cur_limbs
                target_noise_deg = max(CT0.noise_deg, CT1.noise_deg)
        else:
            raise ValueError
        return MetaInfo(target_libms, target_noise_deg)
    elif operation in['homo_mul_scalar_double', 'homo_square']:
        CT0 = metadata[inputs[0]]
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg + 1)
    elif operation in['homo_mul_pt', 'homo_mul']:
        CT0, CT1 = metadata[inputs[0]], metadata[inputs[1]] #this is wrong
        if metadata["RESCALE_TECH"] == "FLEXIBLEAUTO":
            if CT0.cur_limbs > CT1.cur_limbs:
                target_libms = CT1.cur_limbs
                target_noise_deg = CT1.noise_deg
            elif CT0.cur_limbs < CT1.cur_limbs:
                target_libms = CT0.cur_limbs
                target_noise_deg = CT0.noise_deg
            else:
                target_libms = CT0.cur_limbs
                target_noise_deg = max(CT0.noise_deg, CT1.noise_deg)
        else:
            raise ValueError
        return MetaInfo(target_libms, 2)
    elif operation == 'adjust_levels_and_depth':
        CT0, CT1 = metadata[inputs[0]], metadata[inputs[1]] #this is wrong
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg)
    elif operation == 'homo_rescale':
        CT0, scale_level = metadata[inputs[0]], int(inputs[1])
        return MetaInfo(CT0.cur_limbs - scale_level, CT0.noise_deg - scale_level)
    elif operation == 'mod_raise':
        CT0, raise_level = metadata[inputs[0]], int(inputs[1])
        return MetaInfo(raise_level, CT0.noise_deg)
    else:
        return MetaInfo(-1, -1)


def process_graph_topologically(graph, initial_metadata):
    """
    Process the graph in topological order and calculate metadata for each node.
    """
    # Create a copy of the metadata to update
    metadata = initial_metadata.copy()

    # Perform topological sort (from NODE_OUT to NODE_IN)
    topological_order = list(nx.topological_sort(graph))
    
    # Process each node in topological order
    for node in topological_order:
        # If metadata is not already computed for the node, calculate it
        if node not in metadata:
            metainfo = calculate_metadata(graph, metadata, node)
            metadata[node] = metainfo
            print(f"Node: {node}")
            print(f"  data: {metainfo}")
            print()
    
    return metadata

# Example of provided metadata for source nodes (like NODE_IN)
initial_metadata = {
    "RESCALE_TECH" : "FLEXIBLEAUTO",
    "IN_NODE": MetaInfo(2, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][0]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][1]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][2]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][3]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][4]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][5]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][6]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][0]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][1]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][2]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][3]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][4]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][5]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][6]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][0]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][1]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][2]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][3]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][4]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][5]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][6]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][0]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][1]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][2]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][3]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][4]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][5]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][6]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][0]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][1]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][2]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][3]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][4]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][5]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][6]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][0]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][1]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][2]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][3]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][4]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][5]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][6]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][0]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][1]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][2]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][3]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][4]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][5]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][6]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][0]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][1]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][2]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][3]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][4]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][5]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][6]" : MetaInfo(5, 1),
} 

# def assign_metadata(graph, start_node, end_node):
#     # Dictionary to store metadata for each node
#     metadata_dict = {"IN_NODE" : MetaInfo(2, 1)}

#     # Perform a breadth-first search (BFS) to traverse the graph from start_node to end_node
#     queue = [start_node]
#     visited = set()

#     while queue:
#         current_node = queue.pop(0)

#         if current_node == end_node:
#             break  # Stop if we reach NODE_OUT

#         # Skip nodes we've already processed
#         if current_node in visited:
#             continue
#         visited.add(current_node)

#         # Retrieve the operation and inputs for the current node
#         operation = graph.nodes[current_node].get("operation")
#         inputs = graph.nodes[current_node].get("inputs")
        
#         print(current_node)
#         calculate_metadata(operation, inputs, metadata_dict)
#         metadata_dict[current_node] = calculate_metadata(operation, inputs, metadata_dict)

#         # Add successors (dependent nodes) to the queue for BFS traversal
#         print("  Successors: ", end="")
#         for successor in graph.successors(current_node):
#             print(successor, end=" ")
#             queue.append(successor)
#         print()

#     return metadata_dict


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

# Process the graph in topological order and calculate metadata
final_metadata = process_graph_topologically(graph, initial_metadata)

# # Print the metadata for each node in the path
# for node, data in final_metadata.items():
#     print(f"Node: {node}")
#     print(f"  data: {data}")
#     print()
