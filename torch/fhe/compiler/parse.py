import re
import networkx as nx
from pyvis.network import Network

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


def calculate_metadata(graph, metadata, adjust_record, node):
    operation = graph.nodes[node].get("operation")
    inputs = graph.nodes[node].get("inputs")

    if operation in['assignment', 'homo_rotate', 'extract_cv', 'key_switch_P_ext', 'modup_to_ext', 'eval_fast_rotate', 'moddown_from_ext', '_cipher_automorphism', 'homo_add_scalar_double', 'homo_mul_scalar_int', 'assign_scaling_factor']:
        CT0 = metadata[inputs[0]]
        return MetaInfo(CT0.cur_limbs, CT0.noise_deg)
    elif operation in ['homo_add', 'homo_sub', 'adjust_levels_and_depth']:
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
        if not (CT0.cur_limbs == target_libms and CT0.noise_deg == target_noise_deg):
            adjust_record.append((inputs[0], CT0.cur_limbs, CT0.noise_deg, target_libms, target_noise_deg))
            print("DO RESCALE: {} from limb {} noise_deg {} to limb {} noise_deg {}".format(inputs[0], CT0.cur_limbs, CT0.noise_deg, target_libms, target_noise_deg))
        if not (CT1.cur_limbs == target_libms and CT1.noise_deg == target_noise_deg):
            adjust_record.append((inputs[1], CT1.cur_limbs, CT1.noise_deg, target_libms, target_noise_deg))
            print("DO RESCALE: {} from limb {} noise_deg {} to limb {} noise_deg {}".format(inputs[1], CT1.cur_limbs, CT1.noise_deg, target_libms, target_noise_deg))

        return MetaInfo(target_libms, target_noise_deg)
    elif operation in['homo_mul_scalar_double', 'homo_square']:
        CT0 = metadata[inputs[0]]
        if metadata["RESCALE_TECH"] == "FLEXIBLEAUTO":
            if CT0.noise_deg == 2:
                return MetaInfo(CT0.cur_limbs - 1, 2)
            else:
                return MetaInfo(CT0.cur_limbs, 2)
        else:
            raise ValueError

    elif operation in['homo_mul_pt', 'homo_mul']:
        CT0, CT1 = metadata[inputs[0]], metadata[inputs[1]] 
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

            if target_noise_deg == 2:
                target_noise_deg -= 1
                target_libms -= 1
        else:
            raise ValueError
        if not (CT0.cur_limbs == target_libms and CT0.noise_deg == target_noise_deg):
            adjust_record.append((inputs[0], CT0.cur_limbs, CT0.noise_deg, target_libms, target_noise_deg))
            print("DO RESCALE: {} from limb {} noise_deg {} to limb {} noise_deg {}".format(inputs[0], CT0.cur_limbs, CT0.noise_deg, target_libms, target_noise_deg))
        if not (CT1.cur_limbs == target_libms and CT1.noise_deg == target_noise_deg):
            adjust_record.append((inputs[1], CT1.cur_limbs, CT1.noise_deg, target_libms, target_noise_deg))
            print("DO RESCALE: {} from limb {} noise_deg {} to limb {} noise_deg {}".format(inputs[1], CT1.cur_limbs, CT1.noise_deg, target_libms, target_noise_deg))

        return MetaInfo(target_libms, 2)
    # elif operation == 'adjust_levels_and_depth':
    #     CT0, CT1 = metadata[inputs[0]], metadata[inputs[1]] #this is wrong
    #     return MetaInfo(CT0.cur_limbs, CT0.noise_deg)
    elif operation in ['homo_rescale', 'homo_rescale_internal']:
        CT0, scale_level = metadata[inputs[0]], int(inputs[1])
        return MetaInfo(CT0.cur_limbs - scale_level, CT0.noise_deg - scale_level)
    elif operation == 'mod_raise':
        CT0, raise_level = metadata[inputs[0]], int(inputs[1])
        return MetaInfo(raise_level, CT0.noise_deg)
    else:
        print(operation)
        raise ValueError


def process_graph_topologically(graph, initial_metadata):
    """
    Process the graph in topological order and calculate metadata for each node.
    """
    # Create a copy of the metadata to update
    metadata = initial_metadata.copy()
    adjust_record = []
    # Perform topological sort (from NODE_OUT to NODE_IN)
    topological_order = list(nx.topological_sort(graph))
    
    # Process each node in topological order
    for node in topological_order:
        # If metadata is not already computed for the node, calculate it
        if node not in metadata:
            metainfo = calculate_metadata(graph, metadata, adjust_record, node)
            metadata[node] = metainfo
            print(f"Node: {node}")
            print(f"  data: {metainfo}")
            print()
    return metadata, adjust_record

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
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][7]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][8]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][9]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][10]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][11]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][12]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][13]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[0][14]" : MetaInfo(22, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][0]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][1]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][2]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][3]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][4]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][5]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][6]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][7]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][8]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][9]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][10]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][11]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][12]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][13]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[1][14]" : MetaInfo(23, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][0]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][1]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][2]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][3]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][4]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][5]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][6]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][7]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][8]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][9]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][10]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][11]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][12]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][13]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[2][14]" : MetaInfo(24, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][0]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][1]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][2]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][3]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][4]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][5]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][6]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][7]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][8]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][9]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][10]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][11]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][12]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][13]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0hatTPreFFT[3][14]" : MetaInfo(25, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][0]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][1]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][2]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][3]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][4]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][5]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][6]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][7]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][8]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][9]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][10]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][11]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][12]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][13]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[0][14]" : MetaInfo(8, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][0]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][1]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][2]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][3]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][4]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][5]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][6]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][7]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][8]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][9]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][10]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][11]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][12]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][13]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[1][14]" : MetaInfo(7, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][0]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][1]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][2]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][3]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][4]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][5]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][6]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][7]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][8]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][9]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][10]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][11]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][12]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][13]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[2][14]" : MetaInfo(6, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][0]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][1]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][2]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][3]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][4]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][5]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][6]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][7]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][8]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][9]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][10]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][11]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][12]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][13]" : MetaInfo(5, 1),
    "cryptoContext.BsContext.m_U0PreFFT[3][14]" : MetaInfo(5, 1),
} 


def print_graph_info(graph):
    for node in graph.nodes:
        print(f"Node: {node}")
        print(f"  Operation: {graph.nodes[node].get('operation')}")
        print(f"  Inputs: {graph.nodes[node].get('inputs')}")
        print(f"  Outputs: {[n for n in graph.successors(node)]}")
        print()


def draw_graph(graph):
    # Create a chain (path) graph with 10 nodes
    G = graph

    # Compute levels for each node using topological order.
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            levels[node] = 0
        else:
            levels[node] = max(levels[pred] for pred in preds) + 1

    # Group nodes by their level.
    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Compute positions for each node (layered layout).
    pos = {}
    vertical_gap = 150   # vertical spacing between levels
    horizontal_gap = 150 # horizontal spacing within each level
    for level, nodes in level_nodes.items():
        count = len(nodes)
        # Center nodes horizontally on each level.
        start_x = - (count - 1) * horizontal_gap / 2
        for i, node in enumerate(nodes):
            pos[node] = {"x": start_x + i * horizontal_gap, "y": level * vertical_gap}

    # Create a PyVis network.
    net = Network(height="600px", width="100%", notebook=False)

    # Import the entire NetworkX graph (nodes and edges) into PyVis.
    net.from_nx(G)

    def get_color(operation):
        if operation in ['homo_add', 'homo_sub', 'key_switch_P_ext']:
            return "green"
        if operation in ['eval_fast_rotate', 'homo_rotate']:
            return "blue"
        if operation in ['homo_mul_pt', 'homo_mul', 'homo_mul_scalar_double', 'homo_square']:
            return "red"
        if operation in ['homo_rescale', 'homo_rescale_internal', 'adjust_levels_and_depth', 'modup_to_ext', 'moddown_from_ext']:
            return "purple"
        if operation in ['mod_raise', 'assignment', 'extract_cv']:
            return "gray"
        else:
            return "orange"

    # Update each node with computed positions and add custom label info.
    for node in net.nodes:
        node_id = node["id"]
        # Create your custom info string.
        custom_info = node.get("operation")
        # You can choose to append to the existing label or overwrite it.
        # For instance, if the original label is just the node id:
        node["label"] = f"{node['label']}\n{custom_info}"
        # Assign a color based on the level, using the mapping, or a default value.
        node["color"] = get_color(custom_info)

        # Set fixed positions so the physics engine doesn't change them.
        if node_id in pos:
            node["x"] = pos[node_id]["x"]
            node["y"] = pos[node_id]["y"]
            node["fixed"] = {"x": True, "y": True}

    # Disable physics to keep nodes in assigned positions.
    net.toggle_physics(False)

    # Save the interactive graph as an HTML file
    net.show("interactive_topo_graph.html", notebook=False)

def generate_execution_plan(G: nx.DiGraph):
    """
    Generates an execution plan for a DAG.
    Each "line" in the plan is a list of nodes that can be executed concurrently.
    """
    # Work on a copy so the original graph remains intact.
    G_copy = G.copy()
    plan = []
    
    # Continue until there are no nodes left.
    while G_copy.nodes:
        # Find all nodes with no incoming edges (ready to execute).
        ready = [n for n in G_copy.nodes if G_copy.in_degree(n) == 0]
        if not ready:
            raise ValueError("Graph has a cycle or a dependency error!")
        plan.append(ready)
        # Remove these nodes from the graph.
        G_copy.remove_nodes_from(ready)
    
    return plan


        
# Sample code string
with open("sample_code.txt", "r") as f:
    code = f.read()

# Parse the code and construct the graph
graph = parse_code_to_graph(code)

# Print the information about the graph
print_graph_info(graph)

#save graph to pdf
draw_graph(graph)

# Process the graph in topological order and calculate metadata
final_metadata, adjust_record = process_graph_topologically(graph, initial_metadata)

for item in adjust_record:
    print(item)

from collections import Counter
for item, occur_time in Counter(adjust_record).items():
    if occur_time > 1:
        print(item, occur_time)

plan = generate_execution_plan(graph)
for line in plan:
    print(line)

# # Print the metadata for each node in the path
# for node, data in final_metadata.items():
#     print(f"Node: {node}")
#     print(f"  data: {data}")
#     print()
