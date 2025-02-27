import ast
import networkx as nx
import matplotlib.pyplot as plt
import re

# Step 1: Parse the Python Code from a Text File
def parse_python_code(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

# Step 2: Extract Function Calls, Attribute Accesses, and Indexing
def extract_operations(code):
    operations = []
    
    # Regular expression for function calls like 'homo_ops.function_name'
    pattern_function_call = r'(\w+)\.(\w+)\(([^)]*)\)'  # Captures object, function name, and arguments
    matches_function_call = re.finditer(pattern_function_call, code)
    
    for match in matches_function_call:
        object_name = match.group(1)
        function_name = match.group(2)
        arguments = match.group(3).split(',')
        
        # Remove any whitespace around arguments
        arguments = [arg.strip() for arg in arguments]
        
        # Save the function call operation and its arguments
        operations.append({
            'operation': f"{object_name}.{function_name}",
            'arguments': arguments
        })
    
    # Regular expression for attribute accesses, including array/indexing accesses
    pattern_attr_access = r'(\w+)\.(\w+)((?:\[[^\]]*\])*)'  # Captures object, attribute, and indexing
    matches_attr_access = re.finditer(pattern_attr_access, code)
    
    for match in matches_attr_access:
        object_name = match.group(1)
        attribute_name = match.group(2)
        indices = match.group(3)
        
        # Save the attribute access operation
        operations.append({
            'operation': f"{object_name}.{attribute_name}{indices}",
            'arguments': [object_name]  # The object itself is the input for this operation
        })

    return operations

# Step 3: Create a Computation Graph Based on the Operations
def create_computation_graph(operations):
    G = nx.DiGraph()  # Directed graph to represent the computation flow

    # For each operation, add a node and its dependencies
    for operation in operations:
        op_name = operation['operation']
        inputs = operation['arguments']
        
        # Add the node for the operation
        G.add_node(op_name)

        # Create edges from inputs to this operation (input dependencies)
        for input_op in inputs:
            if input_op != 'cryptoContext':  # Ignore 'cryptoContext' as it's not a dependency
                G.add_edge(input_op, op_name)
    
    return G

# Step 4: Save the Computation Graph as a PDF
def save_graph_as_pdf(G, output_pdf_path):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
    plt.title("Computation Graph")
    
    # Save the graph as a PDF
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()

# Step 5: Main Function to Parse the Code and Create Graph
def main():
    # Provide the path to your Python script file
    file_path = 'res.py'  # Replace with the actual path to your Python code file
    output_pdf_path = 'computation_graph.pdf'  # Output PDF file path
    
    code = parse_python_code(file_path)
    
    # Extract operations from the code
    operations = extract_operations(code)
    
    # Create a computation graph from the operations
    G = create_computation_graph(operations)
    
    # Save the computation graph as a PDF
    save_graph_as_pdf(G, output_pdf_path)
    print(f"Computation graph saved to {output_pdf_path}")

if __name__ == "__main__":
    main()
