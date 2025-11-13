"""
Few-Shot Learning for Time Series Analysis

This module implements state-of-the-art few-shot learning techniques for time series
classification, including prototypical networks, LSTM/GRU embeddings, and transformer-based
approaches with comprehensive anomaly detection capabilities.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/timeseries_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TimeSeriesDataGenerator:
    """Generate synthetic time series data for few-shot learning experiments."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the data generator with configuration parameters.

        Args:
            config: Configuration dictionary containing data generation parameters
        """
        self.config = config
        self.sequence_length = config["data"]["sequence_length"]
        self.num_support_samples = config["data"]["num_support_samples"]
        self.num_query_samples = config["data"]["num_query_samples"]
        self.noise_level = config["data"]["noise_level"]
        self.num_classes = config["data"]["num_classes"]

    def generate_sinusoidal_class(
        self, n_samples: int, frequency: float = 1.0, phase: float = 0.0
    ) -> np.ndarray:
        """
        Generate sinusoidal time series with specified parameters.

        Args:
            n_samples: Number of samples to generate
            frequency: Frequency of the sinusoidal wave
            phase: Phase shift of the sinusoidal wave

        Returns:
            Array of shape (n_samples, sequence_length) containing generated time series
        """
        t = np.linspace(0, 3 * np.pi, self.sequence_length)
        base_signal = np.sin(frequency * t + phase)
        
        samples = []
        for _ in range(n_samples):
            noise = self.noise_level * np.random.randn(self.sequence_length)
            sample = base_signal + noise
            samples.append(sample)
        
        return np.array(samples)

    def generate_trend_class(
        self, n_samples: int, trend_slope: float = 0.1
    ) -> np.ndarray:
        """
        Generate time series with linear trend.

        Args:
            n_samples: Number of samples to generate
            trend_slope: Slope of the linear trend

        Returns:
            Array of shape (n_samples, sequence_length) containing generated time series
        """
        t = np.linspace(0, 1, self.sequence_length)
        base_signal = trend_slope * t
        
        samples = []
        for _ in range(n_samples):
            noise = self.noise_level * np.random.randn(self.sequence_length)
            sample = base_signal + noise
            samples.append(sample)
        
        return np.array(samples)

    def generate_seasonal_class(
        self, n_samples: int, seasonal_period: int = 10
    ) -> np.ndarray:
        """
        Generate seasonal time series.

        Args:
            n_samples: Number of samples to generate
            seasonal_period: Period of the seasonal component

        Returns:
            Array of shape (n_samples, sequence_length) containing generated time series
        """
        t = np.linspace(0, 3 * np.pi, self.sequence_length)
        base_signal = np.sin(2 * np.pi * t / seasonal_period)
        
        samples = []
        for _ in range(n_samples):
            noise = self.noise_level * np.random.randn(self.sequence_length)
            sample = base_signal + noise
            samples.append(sample)
        
        return np.array(samples)

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a complete few-shot learning dataset.

        Returns:
            Tuple containing (support_data, support_labels, query_data, query_labels)
        """
        logger.info("Generating few-shot learning dataset...")
        
        # Generate different types of time series for each class
        class_generators = [
            lambda n: self.generate_sinusoidal_class(n, frequency=1.0, phase=0.0),
            lambda n: self.generate_sinusoidal_class(n, frequency=1.0, phase=np.pi/2),
            lambda n: self.generate_trend_class(n, trend_slope=0.1),
            lambda n: self.generate_seasonal_class(n, seasonal_period=10),
        ]
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx in range(self.num_classes):
            generator = class_generators[class_idx % len(class_generators)]
            
            # Generate support set
            support_class = generator(self.num_support_samples)
            support_data.append(support_class)
            support_labels.extend([class_idx] * self.num_support_samples)
            
            # Generate query set
            query_class = generator(self.num_query_samples)
            query_data.append(query_class)
            query_labels.extend([class_idx] * self.num_query_samples)
        
        support_data = np.concatenate(support_data, axis=0)
        query_data = np.concatenate(query_data, axis=0)
        support_labels = np.array(support_labels)
        query_labels = np.array(query_labels)
        
        logger.info(f"Generated dataset: {len(support_data)} support samples, {len(query_data)} query samples")
        
        return support_data, support_labels, query_data, query_labels


class EmbeddingNetwork(nn.Module):
    """Enhanced embedding network with multiple architecture options."""

    def __init__(
        self,
        input_dim: int = 1,
        embedding_dim: int = 32,
        architecture: str = "conv",
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the embedding network.

        Args:
            input_dim: Input dimension (number of channels)
            embedding_dim: Dimension of the output embedding
            architecture: Type of architecture ('conv', 'lstm', 'gru', 'transformer')
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        
        if architecture == "conv":
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif architecture == "lstm":
            self.lstm = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif architecture == "gru":
            self.gru = nn.GRU(input_dim, 64, batch_first=True, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif architecture == "transformer":
            # Ensure d_model is divisible by nhead
            d_model = max(input_dim, 4)  # At least 4 for 4 heads
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=64, dropout=dropout
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(d_model, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        if self.architecture == "conv":
            return self.net(x)
        elif self.architecture == "lstm":
            x = x.transpose(1, 2)  # (batch_size, sequence_length, input_dim)
            _, (h_n, _) = self.lstm(x)
            # Concatenate forward and backward hidden states
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
            return self.fc(h_n)
        elif self.architecture == "gru":
            x = x.transpose(1, 2)  # (batch_size, sequence_length, input_dim)
            _, h_n = self.gru(x)
            # Concatenate forward and backward hidden states
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
            return self.fc(h_n)
        elif self.architecture == "transformer":
            x = x.transpose(1, 2)  # (batch_size, sequence_length, input_dim)
            x = self.input_projection(x)  # Project to d_model if needed
            x = x.transpose(0, 1)  # (sequence_length, batch_size, d_model)
            x = self.transformer(x)
            # Global average pooling
            x = x.mean(dim=0)  # (batch_size, d_model)
            return self.fc(x)


class PrototypicalNetwork:
    """Prototypical network for few-shot learning."""

    def __init__(self, embedding_network: EmbeddingNetwork) -> None:
        """
        Initialize the prototypical network.

        Args:
            embedding_network: Pre-trained embedding network
        """
        self.embedding_network = embedding_network
        self.prototypes: Optional[np.ndarray] = None
        self.class_labels: Optional[np.ndarray] = None

    def compute_prototypes(
        self, support_data: np.ndarray, support_labels: np.ndarray
    ) -> None:
        """
        Compute class prototypes from support set.

        Args:
            support_data: Support set data
            support_labels: Support set labels
        """
        logger.info("Computing class prototypes...")
        
        with torch.no_grad():
            support_tensor = torch.FloatTensor(support_data).unsqueeze(1)
            embeddings = self.embedding_network(support_tensor).numpy()
        
        unique_labels = np.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            class_embeddings = embeddings[support_labels == label]
            prototype = np.mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        self.prototypes = np.array(prototypes)
        self.class_labels = unique_labels
        
        logger.info(f"Computed {len(unique_labels)} prototypes")

    def predict(self, query_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for query data using prototype matching.

        Args:
            query_data: Query data to classify

        Returns:
            Tuple containing (predictions, distances)
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not computed. Call compute_prototypes first.")
        
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_data).unsqueeze(1)
            query_embeddings = self.embedding_network(query_tensor).numpy()
        
        # Compute distances to prototypes
        distances = np.linalg.norm(
            query_embeddings[:, np.newaxis] - self.prototypes[np.newaxis, :], axis=2
        )
        
        # Predict based on nearest prototype
        predictions = self.class_labels[np.argmin(distances, axis=1)]
        
        return predictions, distances


class AnomalyDetector:
    """Anomaly detection using multiple methods."""

    def __init__(self, method: str = "isolation_forest") -> None:
        """
        Initialize the anomaly detector.

        Args:
            method: Anomaly detection method ('isolation_forest', 'autoencoder')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the anomaly detection model.

        Args:
            data: Training data for anomaly detection
        """
        logger.info(f"Training {self.method} anomaly detector...")
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(contamination=0.1, random_state=42)
            data_scaled = self.scaler.fit_transform(data)
            self.model.fit(data_scaled)
        elif self.method == "autoencoder":
            self._train_autoencoder(data)
        
        logger.info("Anomaly detector training completed")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.

        Args:
            data: Data to analyze for anomalies

        Returns:
            Array of anomaly scores (-1 for anomalies, 1 for normal)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit first.")
        
        if self.method == "isolation_forest":
            data_scaled = self.scaler.transform(data)
            return self.model.predict(data_scaled)
        elif self.method == "autoencoder":
            return self._predict_autoencoder(data)

    def _train_autoencoder(self, data: np.ndarray) -> None:
        """Train autoencoder for anomaly detection."""
        # Simplified autoencoder implementation
        # In practice, you'd want a more sophisticated architecture
        pass

    def _predict_autoencoder(self, data: np.ndarray) -> np.ndarray:
        """Predict using autoencoder."""
        # Simplified implementation
        return np.ones(len(data))


class TimeSeriesAnalyzer:
    """Main class for comprehensive time series analysis."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the time series analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_generator = TimeSeriesDataGenerator(self.config)
        self.embedding_network = EmbeddingNetwork(
            embedding_dim=self.config["model"]["embedding_dim"],
            architecture="conv",
            dropout=self.config["model"]["dropout"],
        )
        self.prototypical_network = PrototypicalNetwork(self.embedding_network)
        self.anomaly_detector = AnomalyDetector()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def run_analysis(self) -> Dict:
        """
        Run complete few-shot learning analysis.

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting comprehensive time series analysis...")
        
        # Generate data
        support_data, support_labels, query_data, query_labels = self.data_generator.generate_dataset()
        
        # Compute prototypes
        self.prototypical_network.compute_prototypes(support_data, support_labels)
        
        # Make predictions
        predictions, distances = self.prototypical_network.predict(query_data)
        
        # Calculate metrics
        accuracy = accuracy_score(query_labels, predictions)
        precision = precision_score(query_labels, predictions, average="weighted")
        recall = recall_score(query_labels, predictions, average="weighted")
        f1 = f1_score(query_labels, predictions, average="weighted")
        
        # Anomaly detection
        self.anomaly_detector.fit(support_data)
        anomaly_scores = self.anomaly_detector.predict(query_data)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "predictions": predictions,
            "true_labels": query_labels,
            "distances": distances,
            "anomaly_scores": anomaly_scores,
            "support_data": support_data,
            "query_data": query_data,
            "support_labels": support_labels,
        }
        
        logger.info(f"Analysis completed. Accuracy: {accuracy:.2%}")
        
        return results

    def create_visualizations(self, results: Dict) -> None:
        """
        Create comprehensive visualizations of the analysis results.

        Args:
            results: Analysis results dictionary
        """
        logger.info("Creating visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Query Predictions",
                "Confusion Matrix",
                "Embedding Space Visualization",
                "Anomaly Detection Results"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Query predictions
        query_data = results["query_data"]
        predictions = results["predictions"]
        
        for i in range(min(5, len(query_data))):
            fig.add_trace(
                go.Scatter(
                    y=query_data[i],
                    mode="lines",
                    name=f"Query {i+1} (Pred: {predictions[i]})",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot 2: Confusion matrix
        cm = confusion_matrix(results["true_labels"], results["predictions"])
        fig.add_trace(
            go.Heatmap(
                z=cm,
                colorscale="Blues",
                showscale=True
            ),
            row=1, col=2
        )
        
        # Plot 3: Embedding space (simplified 2D projection)
        # In practice, you'd use t-SNE or UMAP for better visualization
        distances = results["distances"]
        fig.add_trace(
            go.Scatter(
                x=distances[:, 0],
                y=distances[:, 1],
                mode="markers",
                marker=dict(
                    color=results["true_labels"],
                    colorscale="viridis",
                    size=8
                ),
                name="Query Points"
            ),
            row=2, col=1
        )
        
        # Plot 4: Anomaly detection
        anomaly_scores = results["anomaly_scores"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(anomaly_scores))),
                y=anomaly_scores,
                mode="markers",
                marker=dict(
                    color=anomaly_scores,
                    colorscale="RdYlBu_r",
                    size=8
                ),
                name="Anomaly Scores"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Few-Shot Time Series Learning Analysis Results",
            showlegend=True
        )
        
        # Save plot
        fig.write_html("notebooks/analysis_results.html")
        fig.show()
        
        logger.info("Visualizations created and saved")


def main() -> None:
    """Main function to run the analysis."""
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Create visualizations
    analyzer.create_visualizations(results)
    
    # Print results
    print("\n" + "="*50)
    print("FEW-SHOT TIME SERIES LEARNING RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1-Score: {results['f1_score']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
