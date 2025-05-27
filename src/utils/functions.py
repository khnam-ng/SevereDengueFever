"""
Utility functions for the project.

author: Vetivert? 💐 
created: 14/04/2025 @ 17:34:36
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    classification_report
)

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

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, valid_scores, _, _ = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=cv)
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.2, color="r")
    plt.fill_between(train_sizes, valid_mean - valid_std,
                     valid_mean + valid_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, valid_mean, 'o-', color="g", label="Validation score")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Model's Learning Curve")
    plt.show()

def get_feature_importance(model, importance_type='gain'):
    """
    Get feature importance.
    
    Args:
        importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        
    Returns:
        DataFrame with feature importances
    """
    
    # Get feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df
    
def plot_feature_importance(model, x_features, top_n=10, importance_type='gain'):
    """
    Plot feature importance.
    
    Args:
        top_n: Number of top features to show
        importance_type: Type of importance metric
    """

    importance_df = model.get_booster().get_score(importance_type=importance_type)
    
    mapped_importances = {x_features[int(k[1:])]: v for k, v in importance_df.items()}

    mapped_importances_df = pd.DataFrame({
        'Feature': list(mapped_importances.keys()),
        'Importance': list(mapped_importances.values())
    }).sort_values(by='Importance', ascending=False)


    sns.barplot(x='Importance', y='Feature', data=mapped_importances_df.head(top_n))
    plt.title("Top 10 Important Features (XGBoost)")
    plt.axvline(x=np.mean(mapped_importances_df['Importance']).item(), color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    return mapped_importances_df
    
def evaluate(model, X, y_true, model_type='classifier'):
    """
    Evaluate model performance.
    
    Args:
        X: Features
        y_true: True target values
        
    Returns:
        Dict of evaluation metrics
    """
    y_pred_proba = model.predict(X)
    y_pred = [1 if float(p) >= 0.5 else 0 for p in y_pred_proba]
    y_true = [1 if float(p) >= 0.5 else 0 for p in y_true]
    # super().evaluate(X, y_true)  # Call the base class evaluate method
    
    if model_type == 'classifier':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0), #average='weighted'
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_prob = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                pass
    else:
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.rename(index={'0': 'non dangerous', '1': 'dangerous'}, inplace=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:3, :3], annot=True, cmap='Blues', fmt=".2f")  # Only show precision, recall, f1-score
    plt.title('Classification Report')
    plt.ylabel('Class')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.show()

    return metrics