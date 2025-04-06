# Import the graphviz library
import graphviz
import os

# Ensure Graphviz executables are in the system's PATH or specify the path
# Example (if needed, adjust the path to your Graphviz installation):
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Create a new directed graph
dot = graphviz.Digraph('NCF_Workflow_Simple_LR_WiderNodes', comment='script workflow')

# --- Set graph orientation and spacing ---
# rankdir='LR' for Left-to-Right orientation
# ranksep='0.5' reduces the space between columns (ranks)
dot.attr(rankdir='LR', ranksep='0.5', label='script workflow', fontsize='18')

# Define node styles - ADDING width hints here
node_width = '1.8' # Increase this value to make nodes wider (in inches)
process_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue', 'width': node_width}
decision_attrs = {'shape': 'diamond', 'style': 'filled', 'fillcolor': 'yellow', 'width': node_width}
# Start/End nodes might look better without forced width, but you could add it:
# start_end_attrs = {'shape': 'Mdiamond', 'style': 'filled', 'fillcolor': 'palegreen', 'width': node_width}
start_end_attrs = {'shape': 'Mdiamond', 'style': 'filled', 'fillcolor': 'palegreen'} # Keeping start/end default width for now

# --- Nodes ---
# Nodes will now use the width defined in their respective attribute dictionaries

dot.node('start', 'Start Script', start_end_attrs)
dot.node('setup', 'Setup (Device)', process_attrs)

# Data Handling Stage
dot.node('ask_preprocess', 'Preprocess Data\n(Incl. optional .dat conversion)?', decision_attrs)
dot.node('run_preprocess', 'Convert & Preprocess Data\n(.dat -> train/val/test.csv)', process_attrs)
dot.node('load_preprocessed', 'Load Preprocessed Data\n(train/val/test.csv)', process_attrs)

# Training Stage
dot.node('ask_train', 'Train Model?', decision_attrs)
dot.node('run_train', 'Train NCF Model\n(train/val data -> ncf_model.pt)', process_attrs)

# Evaluation Stage
dot.node('ask_evaluate', 'Evaluate Model?', decision_attrs)
dot.node('run_evaluate', 'Evaluate Model\n(ncf_model.pt + test data -> Metrics)', process_attrs)

dot.node('end', 'End Script', start_end_attrs)


# --- Edges (Workflow) ---
# Edges remain the same

dot.edge('start', 'setup')
dot.edge('setup', 'ask_preprocess')

# Data Path
dot.edge('ask_preprocess', 'run_preprocess', label='Yes')
dot.edge('ask_preprocess', 'load_preprocessed', label='No')
dot.edge('run_preprocess', 'ask_train')
dot.edge('load_preprocessed', 'ask_train')

# Training Path
dot.edge('ask_train', 'run_train', label='Yes')
dot.edge('ask_train', 'ask_evaluate', label='No')
dot.edge('run_train', 'ask_evaluate')

# Evaluation Path
dot.edge('ask_evaluate', 'run_evaluate', label='Yes')
dot.edge('ask_evaluate', 'end', label='No')
dot.edge('run_evaluate', 'end')


# --- Render graph ---
# Saves the graph with a new name
# view=True tries to open the generated image automatically
try:
    output_filename = 'NCF_Workflow_Simple_LR_WiderNodes'
    dot.render(output_filename, view=True, format='png', cleanup=True)
    print(f"Wider Nodes Left-to-Right Graphviz diagram '{output_filename}.png' generated successfully.")
    print("If the image didn't open automatically, check the current directory.")
except graphviz.backend.execute.ExecutableNotFound:
    print("ERROR: Graphviz executable not found.")
    print("Please ensure Graphviz is installed and its 'bin' directory is in your system's PATH.")
    print("You can download Graphviz from: https://graphviz.org/download/")
except Exception as e:
    print(f"An error occurred during graph rendering: {e}")