import matplotlib.pyplot as plt

class ModelVisualizer:
    def __init__(self, model_name):
        """
        Initialize the visualizer with the model name.
        Args:
            model_name (str): Name of the model (e.g., 'IsolationForest', 'GRU').
        """
        self.model_name = model_name

    def plot_isolation_forest(self, data, x_col, y_col, label_col='anomaly'):
        """
        Scatter plot for visualizing Isolation Forest results.
        Args:
            data (pd.DataFrame): Dataframe containing the data.
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            label_col (str): Column name for anomaly labels (-1 = anomaly, 1 = normal).
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x_col], data[y_col], c=data[label_col], cmap='coolwarm', alpha=0.7)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Anomalies Detected by {self.model_name}')
        plt.colorbar(label='Anomaly (-1 = anomaly, 1 = normal)')
        plt.show()

    def plot_gru(self, predictions, ground_truth, timestamps):
        """
        Line plot for GRU predictions vs. ground truth.
        Args:
            predictions (list): Model predictions.
            ground_truth (list): True labels.
            timestamps (list): Corresponding timestamps.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, ground_truth, label='Ground Truth', alpha=0.7)
        plt.plot(timestamps, predictions, label='Predictions', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'GRU Model Predictions vs Ground Truth')
        plt.legend()
        plt.show()

    def plot_lstm_rnn(self, data, anomaly_scores, threshold):
        """
        Histogram for LSTM+RNN anomaly scores.
        Args:
            data (pd.DataFrame): Dataframe containing data.
            anomaly_scores (list): Calculated anomaly scores.
            threshold (float): Threshold for anomaly detection.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=30, color='skyblue', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Anomaly Scores - {self.model_name}')
        plt.legend()
        plt.show()

    def visualize(self, data, **kwargs):
        """
        General method to visualize results based on model type.
        Args:
            data (pd.DataFrame): Dataframe containing the data.
            kwargs: Additional parameters for the specific visualization method.
        """
        if self.model_name == 'IsolationForest':
            self.plot_isolation_forest(data, **kwargs)
        elif self.model_name == 'GRU':
            self.plot_gru(**kwargs)
        elif self.model_name == 'LSTM+RNN':
            self.plot_lstm_rnn(**kwargs)
        else:
            raise ValueError(f"Visualization for {self.model_name} is not implemented.")
