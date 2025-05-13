"""
author: Vetivert? 💐 
created: 17/04/2025 @ 15:00:59
"""
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, GridSearchCV # type: ignore
from sklearn.preprocessing import StandardScaler, label_binarize # type: ignore
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score, 
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from scipy.stats import randint, uniform

import xgboost as xgb
from xgboost import XGBClassifier

from typing import Dict, Any, Union, Tuple, Optional, List



class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_type: str = 'classifier', random_state: int = 66):
        """
        Initialize base model.
        
        Args:
            model_type: Type of model ('classifier' or 'regressor')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Target values
        
        Returns:
            self: The fitted model
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions (classification only).
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        if self.model_type != 'classifier':
            raise ValueError("predict_proba is only available for classifiers")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y_true: True target values
            
        Returns:
            Dict of evaluation metrics
        """
        y_pred = self.predict(X)
        
        if self.model_type == 'classifier':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    y_prob = self.predict_proba(X)[:, 1]
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
        df_report.rename(index={'1': 'non dangerous', '2': 'dangerous'}, inplace=True)
    
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_report.iloc[:3, :3], annot=True, cmap='Blues', fmt=".2f")  # Only show precision, recall, f1-score
        plt.title('Classification Report')
        plt.ylabel('Class')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.show()

        return metrics
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(
        self, 
        model_type: str = 'classifier',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float, int] = 'sqrt',
        random_state: int = 66,
        **kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            model_type: 'classifier' or 'regressor'
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in a leaf
            max_features: Feature selection strategy
            random_state: Random seed
            **kwargs: Additional parameters for RandomForest
        """
        super().__init__(model_type, random_state)
        
        # Common params for both classifier and regressor
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
            **kwargs
        }
        
        # Initialize the model
        if model_type == 'classifier':
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
    
    def fit(self, X, y):
        """
        Fit Random Forest model.
        
        Args:
            X: Training features
            y: Target values
        
        Returns:
            self: The fitted model
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def get_feature_importance(self, X):
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if hasattr(X, 'columns'):  # If X was a DataFrame
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
    
    def plot_feature_importance(self, X, top_n: int = 10):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
        """
        importance_df = self.get_feature_importance(X)
        
        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_df['Feature'][:top_n][::-1], 
            importance_df['Importance'][:top_n][::-1]
        )
        plt.axvline(x=np.mean(importance_df['Importance']), color='red', linestyle='--')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def optimize_hyperparameters(
        self, 
        X, 
        y, 
        param_grid: Dict[str, List], 
        cv: int = 10,
        scoring: str = None,
        search_method: str = 'grid',
        n_iter: int = 10,
        verbose: int = 1
    ):
        """
        Optimize hyperparameters using GridSearch or RandomizedSearch.
        
        Args:
            X: Training features
            y: Target values
            param_grid: Dictionary of parameter grid
            cv: Number of cross-validation folds
            scoring: Scoring metric ('accuracy', 'neg_mean_squared_error', etc.)
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            verbose: Verbosity level
            
        Returns:
            self: Model with optimized hyperparameters
        """
        if scoring is None:
            scoring = 'accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error'
        
        # Create new model instance with the same model_type
        if self.model_type == 'classifier':
            base_model = RandomForestClassifier(random_state=self.random_state)
        else:
            base_model = RandomForestRegressor(random_state=self.random_state)
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        search.fit(X, y)
        
        # Update model with best estimator
        self.model = search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return self


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(
        self, 
        model_type: str = 'classifier',
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        min_child_weight: float = 1,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 66,
        **kwargs
    ):
        """
        Initialize XGBoost model.
        
        Args:
            model_type: 'classifier' or 'regressor'
            learning_rate: Learning rate
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns per tree
            gamma: Minimum loss reduction for partition
            min_child_weight: Minimum sum of instance weight in a child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            **kwargs: Additional parameters for XGBoost
        """
        super().__init__(model_type, random_state)
        
        # Common params
        self.params = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            **kwargs
        }
        
        # Initialize model based on type
        if model_type == 'classifier':
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',  # Default for binary classification
                **self.params
            )
        else:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',  # Default for regression
                **self.params
            )
    
    def fit(self, X, y, eval_set=None, verbose=True): #early_stopping_rounds=None,
        """
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Target values
            eval_set: Validation data for early stopping
            early_stopping_rounds: Stop if validation score doesn't improve
            verbose: Whether to print progress
            
        Returns:
            self: The fitted model
        """
        self.model.fit(
            X, y,
            eval_set=eval_set,
            # early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        self.is_fitted = True
        return self
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Get feature importance
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, x_features, top_n=10, importance_type='gain'):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            importance_type: Type of importance metric
        """
    
        importance_df = self.model.get_booster().get_score(importance_type=importance_type)
        
        mapped_importances = {x_features[int(k[1:])]: v for k, v in importance_df.items()}

        mapped_importances_df = pd.DataFrame({
            'Feature': list(mapped_importances.keys()),
            'Importance': list(mapped_importances.values())
        }).sort_values(by='Importance', ascending=False)


        sns.barplot(x='Importance', y='Feature', data=mapped_importances_df.head(top_n))
        plt.title("Top 10 Important Features (XGBoost)")
        plt.axvline(x=np.mean(mapped_importances_df['Importance']), color='red', linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def optimize_hyperparameters(
        self, 
        X, 
        y, 
        param_grid: Dict[str, List], 
        cv: int = 10,
        scoring: str = None,
        search_method: str = 'grid',
        n_iter: int = 10,
        early_stopping_rounds: int = 50,
        verbose: int = 1
    ):
        """
        Optimize hyperparameters using GridSearch or RandomizedSearch.
        
        Args:
            X: Training features
            y: Target values
            param_grid: Dictionary of parameter grid
            cv: Number of cross-validation folds
            scoring: Scoring metric
            search_method: 'grid' or 'random'
            n_iter: Number of iterations for random search
            early_stopping_rounds: Stop if validation score doesn't improve
            verbose: Verbosity level
            
        Returns:
            self: Model with optimized hyperparameters
        """
        if scoring is None:
            scoring = 'accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error'
        
        # Create base model
        if self.model_type == 'classifier':
            base_model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            base_model = xgb.XGBRegressor(
                random_state=self.random_state,
                eval_metric='rmse'
            )
            
        # Add fit_params for early stopping
        # fit_params = {
        #     'early_stopping_rounds': early_stopping_rounds,
        #     'verbose': False
        # }
            
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                verbose=verbose,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Fit search
        search.fit(X, y)
        
        # Update model with best estimator
        self.model = search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return self

    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y_true: True target values
            
        Returns:
            Dict of evaluation metrics
        """
        y_pred_proba = self.predict(X)
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_proba]
        y_true = [1 if p >= 0.5 else 0 for p in y_true]
        # super().evaluate(X, y_true)  # Call the base class evaluate method
        
        if self.model_type == 'classifier':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    y_prob = self.predict_proba(X)[:, 1]
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

# if __name__ == "__main__":
#     # Sample code to demonstrate usage
#     from sklearn.datasets import load_breast_cancer
#     from sklearn.model_selection import train_test_split
    
#     # Load sample dataset
#     data = load_breast_cancer()
#     X, y = data.data, data.target
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=66
#     )
    
#     # Random Forest example
#     print("Training Random Forest model...")
#     rf_model = RandomForestModel(n_estimators=100, random_state=66)
#     rf_model.fit(X_train, y_train)
#     rf_metrics = rf_model.evaluate(X_test, y_test)
#     print("RF Performance:", rf_metrics)
    
#     # XGBoost example
#     print("\nTraining XGBoost model...")
#     xgb_model = XGBoostModel(n_estimators=100, random_state=66)
#     xgb_model.fit(X_train, y_train)
#     xgb_metrics = xgb_model.evaluate(X_test, y_test)
#     print("XGBoost Performance:", xgb_metrics)