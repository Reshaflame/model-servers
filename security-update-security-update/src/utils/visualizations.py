import matplotlib.pyplot as plt
import os
import pandas as pd

class ModelVisualizer:
    def __init__(self, model_name):
        """
        Initialize the visualizer with the model name.
        Args:
            model_name (str): Name of the model (e.g., 'IsolationForest', 'GRU').
        """
        self.model_name = model_name

    def map_categories_to_integers(self, data, col):
        """
        Map categorical column to integers for visualization.
        Args:
            data (pd.DataFrame): The dataset.
            col (str): The column to map.
        Returns:
            pd.Series: A Series with integer values.
        """
        if pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            print(f"Mapping categorical column {col} to numeric values.")
            return data[col].astype('category').cat.codes
        return data[col]

    def plot_categorical_relationship(self, data, x_col, y_col, label_col='anomaly', title=None, save_dir='data/visualizations'):
        """
        Visualize categorical relationships by enumerating categories.
        Args:
            data (pd.DataFrame): Dataframe containing the data.
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            label_col (str): Column name for anomaly labels (-1 = anomaly, 1 = normal).
            title (str): Title of the plot.
            save_dir (str): Directory to save the plot as an image.
        """
        # Map categorical columns to integers
        data['x_numeric'] = self.map_categories_to_integers(data, x_col)
        data['y_numeric'] = self.map_categories_to_integers(data, y_col)

        # Subsample for clarity if the dataset is too large
        if len(data) > 10000:
            data = data.sample(10000, random_state=42)
            print("Subsampled to 10,000 points for visualization.")

        # Inspect mapped ranges
        print(f"Range of {x_col} (mapped): Min = {data['x_numeric'].min()}, Max = {data['x_numeric'].max()}")
        print(f"Range of {y_col} (mapped): Min = {data['y_numeric'].min()}, Max = {data['y_numeric'].max()}")

        # Prepare the plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            data['x_numeric'],
            data['y_numeric'],
            c=data[label_col],
            cmap='coolwarm',
            alpha=0.6
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title if title else f'Anomalies Detected by {self.model_name}')
        plt.colorbar(scatter, label='Anomaly (-1 = anomaly, 1 = normal)')
        plt.grid(True)

        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{self.model_name}_categorical_relationship.png')
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_isolation_forest(self, data, x_col, y_col, label_col='anomaly', title=None, save_dir='data/visualizations'):
        """
        Scatter plot for visualizing Isolation Forest results with dynamic scale adjustments.
        Args:
            data (pd.DataFrame): Dataframe containing the data.
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            label_col (str): Column name for anomaly labels (-1 = anomaly, 1 = normal).
            title (str): Title of the plot.
            save_dir (str): Directory to save the plot as an image.
        """
        if x_col not in data.columns or y_col not in data.columns or label_col not in data.columns:
            raise ValueError(f"Columns {x_col}, {y_col}, or {label_col} not found in the dataset.")

        # Inspect data ranges
        print(f"Range of {x_col}: Min = {data[x_col].min()}, Max = {data[x_col].max()}")
        print(f"Range of {y_col}: Min = {data[y_col].min()}, Max = {data[y_col].max()}")

        # Handle extreme values by setting quantile-based limits
        x_min, x_max = data[x_col].quantile(0.01), data[x_col].quantile(0.99)
        y_min, y_max = data[y_col].quantile(0.01), data[y_col].quantile(0.99)

        # Prepare the plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            data[x_col],
            data[y_col],
            c=data[label_col],
            cmap='coolwarm',
            alpha=0.7
        )
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title if title else f'Anomalies Detected by {self.model_name}')
        plt.colorbar(scatter, label='Anomaly (-1 = anomaly, 1 = normal)')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)

        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{self.model_name}_anomalies_plot.png')
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def visualize(self, data, plot_type='numerical', **kwargs):
        """
        General method to visualize results based on model type and plot type.
        Args:
            data (pd.DataFrame): Dataframe containing the data.
            plot_type (str): Type of plot ('numerical' or 'categorical').
            kwargs: Additional parameters for the specific visualization method.
        """
        if plot_type == 'numerical':
            self.plot_isolation_forest(data, **kwargs)
        elif plot_type == 'categorical':
            self.plot_categorical_relationship(data, **kwargs)
        else:
            raise ValueError(f"Plot type {plot_type} is not recognized.")
