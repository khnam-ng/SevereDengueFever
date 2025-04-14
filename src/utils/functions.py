"""
Utility functions for the project.

author: Vetivert? 💐 
created: 14/04/2025 @ 17:34:36
"""

# Extract split values from each tree
def extract_split_values(forest, feature_names):
    split_values = {name: [] for name in feature_names}
    
    # Loop through all trees in the forest
    for tree in forest.estimators_:
        # Get tree structure
        tree_struct = tree.tree_
        feature = tree_struct.feature
        threshold = tree_struct.threshold
        
        # Loop through nodes
        for node_id in range(tree_struct.node_count):
            # Check if it's not a leaf
            if tree_struct.children_left[node_id] != tree_struct.children_right[node_id]:
                # Get feature being split on
                feature_id = feature[node_id]
                if feature_id != -2:  # -2 indicates a leaf node
                    feature_name = feature_names[feature_id]
                    split_values[feature_name].append(threshold[node_id])
    
    return split_values

