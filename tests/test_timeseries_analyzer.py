"""
Unit tests for the Few-Shot Time Series Learning project.

This module contains comprehensive tests for all major components including
data generation, model architectures, prototypical networks, and anomaly detection.
"""

import unittest
import numpy as np
import torch
import yaml
from pathlib import Path
import tempfile
import os

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from timeseries_analyzer import (
    TimeSeriesDataGenerator,
    EmbeddingNetwork,
    PrototypicalNetwork,
    AnomalyDetector,
    TimeSeriesAnalyzer
)


class TestTimeSeriesDataGenerator(unittest.TestCase):
    """Test cases for TimeSeriesDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "data": {
                "sequence_length": 50,
                "num_support_samples": 5,
                "num_query_samples": 10,
                "noise_level": 0.1,
                "num_classes": 2
            }
        }
        self.generator = TimeSeriesDataGenerator(self.config)

    def test_initialization(self):
        """Test proper initialization of data generator."""
        self.assertEqual(self.generator.sequence_length, 50)
        self.assertEqual(self.generator.num_support_samples, 5)
        self.assertEqual(self.generator.num_query_samples, 10)
        self.assertEqual(self.generator.noise_level, 0.1)
        self.assertEqual(self.generator.num_classes, 2)

    def test_generate_sinusoidal_class(self):
        """Test sinusoidal class generation."""
        samples = self.generator.generate_sinusoidal_class(3, frequency=1.0, phase=0.0)
        
        self.assertEqual(samples.shape, (3, 50))
        self.assertIsInstance(samples, np.ndarray)
        
        # Check that samples are different due to noise
        self.assertFalse(np.allclose(samples[0], samples[1]))

    def test_generate_trend_class(self):
        """Test trend class generation."""
        samples = self.generator.generate_trend_class(3, trend_slope=0.1)
        
        self.assertEqual(samples.shape, (3, 50))
        self.assertIsInstance(samples, np.ndarray)

    def test_generate_seasonal_class(self):
        """Test seasonal class generation."""
        samples = self.generator.generate_seasonal_class(3, seasonal_period=10)
        
        self.assertEqual(samples.shape, (3, 50))
        self.assertIsInstance(samples, np.ndarray)

    def test_generate_dataset(self):
        """Test complete dataset generation."""
        support_data, support_labels, query_data, query_labels = self.generator.generate_dataset()
        
        # Check shapes
        expected_support_samples = 2 * 5  # 2 classes * 5 samples each
        expected_query_samples = 2 * 10  # 2 classes * 10 samples each
        
        self.assertEqual(support_data.shape, (expected_support_samples, 50))
        self.assertEqual(query_data.shape, (expected_query_samples, 50))
        self.assertEqual(len(support_labels), expected_support_samples)
        self.assertEqual(len(query_labels), expected_query_samples)
        
        # Check labels
        self.assertTrue(np.all(np.isin(support_labels, [0, 1])))
        self.assertTrue(np.all(np.isin(query_labels, [0, 1])))


class TestEmbeddingNetwork(unittest.TestCase):
    """Test cases for EmbeddingNetwork class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 1
        self.sequence_length = 50
        self.embedding_dim = 32

    def test_conv_architecture(self):
        """Test convolutional architecture."""
        model = EmbeddingNetwork(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            architecture="conv"
        )
        
        # Test forward pass
        x = torch.randn(4, self.input_dim, self.sequence_length)
        output = model(x)
        
        self.assertEqual(output.shape, (4, self.embedding_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_lstm_architecture(self):
        """Test LSTM architecture."""
        model = EmbeddingNetwork(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            architecture="lstm"
        )
        
        # Test forward pass
        x = torch.randn(4, self.input_dim, self.sequence_length)
        output = model(x)
        
        self.assertEqual(output.shape, (4, self.embedding_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_gru_architecture(self):
        """Test GRU architecture."""
        model = EmbeddingNetwork(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            architecture="gru"
        )
        
        # Test forward pass
        x = torch.randn(4, self.input_dim, self.sequence_length)
        output = model(x)
        
        self.assertEqual(output.shape, (4, self.embedding_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_transformer_architecture(self):
        """Test transformer architecture."""
        model = EmbeddingNetwork(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            architecture="transformer"
        )
        
        # Test forward pass
        x = torch.randn(4, self.input_dim, self.sequence_length)
        output = model(x)
        
        self.assertEqual(output.shape, (4, self.embedding_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_invalid_architecture(self):
        """Test invalid architecture raises error."""
        with self.assertRaises(ValueError):
            EmbeddingNetwork(architecture="invalid")


class TestPrototypicalNetwork(unittest.TestCase):
    """Test cases for PrototypicalNetwork class."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_network = EmbeddingNetwork(embedding_dim=32, architecture="conv")
        self.prototypical_network = PrototypicalNetwork(self.embedding_network)
        
        # Generate test data
        self.support_data = np.random.randn(10, 50)
        self.support_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.query_data = np.random.randn(6, 50)
        self.query_labels = np.array([0, 0, 0, 1, 1, 1])

    def test_compute_prototypes(self):
        """Test prototype computation."""
        self.prototypical_network.compute_prototypes(self.support_data, self.support_labels)
        
        self.assertIsNotNone(self.prototypical_network.prototypes)
        self.assertIsNotNone(self.prototypical_network.class_labels)
        
        # Check prototype shape
        self.assertEqual(self.prototypical_network.prototypes.shape, (2, 32))
        
        # Check class labels
        self.assertTrue(np.array_equal(self.prototypical_network.class_labels, [0, 1]))

    def test_predict_without_prototypes(self):
        """Test prediction without computed prototypes raises error."""
        with self.assertRaises(ValueError):
            self.prototypical_network.predict(self.query_data)

    def test_predict_with_prototypes(self):
        """Test prediction with computed prototypes."""
        self.prototypical_network.compute_prototypes(self.support_data, self.support_labels)
        predictions, distances = self.prototypical_network.predict(self.query_data)
        
        # Check output shapes
        self.assertEqual(len(predictions), 6)
        self.assertEqual(distances.shape, (6, 2))
        
        # Check predictions are valid class labels
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.anomaly_detector = AnomalyDetector(method="isolation_forest")
        self.data = np.random.randn(100, 50)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.anomaly_detector.method, "isolation_forest")
        self.assertIsNone(self.anomaly_detector.model)

    def test_fit(self):
        """Test model fitting."""
        self.anomaly_detector.fit(self.data)
        
        self.assertIsNotNone(self.anomaly_detector.model)
        self.assertIsNotNone(self.anomaly_detector.scaler)

    def test_predict_without_fit(self):
        """Test prediction without fitting raises error."""
        with self.assertRaises(ValueError):
            self.anomaly_detector.predict(self.data)

    def test_predict_with_fit(self):
        """Test prediction after fitting."""
        self.anomaly_detector.fit(self.data)
        predictions = self.anomaly_detector.predict(self.data)
        
        # Check output shape
        self.assertEqual(len(predictions), 100)
        
        # Check predictions are valid (-1 or 1)
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Test cases for TimeSeriesAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        config = {
            "data": {
                "sequence_length": 30,
                "num_support_samples": 3,
                "num_query_samples": 5,
                "noise_level": 0.05,
                "num_classes": 2
            },
            "model": {
                "embedding_dim": 16,
                "dropout": 0.1
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test proper initialization."""
        analyzer = TimeSeriesAnalyzer(self.config_path)
        
        self.assertIsNotNone(analyzer.config)
        self.assertIsNotNone(analyzer.data_generator)
        self.assertIsNotNone(analyzer.embedding_network)
        self.assertIsNotNone(analyzer.prototypical_network)
        self.assertIsNotNone(analyzer.anomaly_detector)

    def test_run_analysis(self):
        """Test complete analysis run."""
        analyzer = TimeSeriesAnalyzer(self.config_path)
        results = analyzer.run_analysis()
        
        # Check that all expected keys are present
        expected_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "predictions", "true_labels", "distances", "anomaly_scores",
            "support_data", "query_data", "support_labels"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check metric values are valid
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["precision"], 0.0)
        self.assertLessEqual(results["precision"], 1.0)
        self.assertGreaterEqual(results["recall"], 0.0)
        self.assertLessEqual(results["recall"], 1.0)
        self.assertGreaterEqual(results["f1_score"], 0.0)
        self.assertLessEqual(results["f1_score"], 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        config = {
            "data": {
                "sequence_length": 20,
                "num_support_samples": 2,
                "num_query_samples": 3,
                "noise_level": 0.1,
                "num_classes": 2
            },
            "model": {
                "embedding_dim": 16,
                "dropout": 0.1
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        analyzer = TimeSeriesAnalyzer(self.config_path)
        results = analyzer.run_analysis()
        
        # Verify the pipeline produces reasonable results
        self.assertGreater(results["accuracy"], 0.0)
        
        # Check data shapes
        self.assertEqual(len(results["support_data"]), 4)  # 2 classes * 2 samples
        self.assertEqual(len(results["query_data"]), 6)   # 2 classes * 3 samples
        
        # Check predictions match query data length
        self.assertEqual(len(results["predictions"]), len(results["query_data"]))


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTimeSeriesDataGenerator,
        TestEmbeddingNetwork,
        TestPrototypicalNetwork,
        TestAnomalyDetector,
        TestTimeSeriesAnalyzer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
