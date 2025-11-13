# Few-Shot Learning for Time Series Analysis

A comprehensive Python project implementing state-of-the-art few-shot learning techniques for time series classification, featuring prototypical networks, multiple neural architectures, and anomaly detection capabilities.

## Features

- **Multiple Neural Architectures**: Convolutional networks, LSTM, GRU, and Transformer-based embeddings
- **Prototypical Networks**: Few-shot learning using prototype-based classification
- **Anomaly Detection**: Isolation Forest and Autoencoder-based anomaly detection
- **Interactive Web Interface**: Streamlit-based dashboard for exploration and analysis
- **Comprehensive Visualizations**: Plotly-powered interactive plots and charts
- **Synthetic Data Generation**: Realistic time series data with trends, seasonality, and noise
- **Modern Python Practices**: Type hints, comprehensive docstrings, and PEP8 compliance
- **Extensive Testing**: Unit tests covering all major components
- **Configuration Management**: YAML-based configuration system
- **Logging**: Comprehensive logging for debugging and monitoring

## Project Structure

```
├── src/                          # Source code
│   ├── timeseries_analyzer.py    # Main analysis module
│   └── streamlit_app.py         # Streamlit web interface
├── tests/                        # Unit tests
│   └── test_timeseries_analyzer.py
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                        # Data storage
├── models/                      # Model checkpoints
├── notebooks/                   # Jupyter notebooks and outputs
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Few-Shot-Learning-for-Time-Series-Analysis.git
cd Few-Shot-Learning-for-Time-Series-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the main analysis:

```bash
python src/timeseries_analyzer.py
```

### Web Interface

Launch the Streamlit dashboard:

```bash
streamlit run src/streamlit_app.py
```

The web interface will be available at `http://localhost:8501`

### Configuration

Modify `config/config.yaml` to customize:

- Data generation parameters (sequence length, noise level, number of classes)
- Model architecture settings (embedding dimension, dropout rate)
- Training parameters (learning rate, batch size, epochs)
- Visualization settings (figure size, color palette)

### Example Configuration

```yaml
data:
  sequence_length: 50
  num_support_samples: 5
  num_query_samples: 10
  noise_level: 0.1
  num_classes: 2

model:
  embedding_dim: 32
  conv_channels: [16, 32]
  kernel_size: 3
  dropout: 0.1
```

## Key Components

### TimeSeriesDataGenerator

Generates synthetic time series data with various patterns:

- **Sinusoidal**: Sine and cosine waves with different frequencies and phases
- **Trend**: Linear trends with configurable slopes
- **Seasonal**: Periodic patterns with customizable periods
- **Noise**: Configurable Gaussian noise levels

### EmbeddingNetwork

Multiple neural architecture options:

- **Convolutional**: 1D CNN with batch normalization and dropout
- **LSTM**: Bidirectional LSTM with attention mechanisms
- **GRU**: Bidirectional GRU for efficient sequence modeling
- **Transformer**: Multi-head attention-based architecture

### PrototypicalNetwork

Implements few-shot learning using prototype-based classification:

1. Compute embeddings for support set samples
2. Calculate class prototypes (mean embeddings per class)
3. Classify query samples based on distance to prototypes

### AnomalyDetector

Multiple anomaly detection methods:

- **Isolation Forest**: Unsupervised anomaly detection
- **Autoencoder**: Reconstruction-based anomaly detection (planned)

## Running Tests

Execute the test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python tests/test_timeseries_analyzer.py
```

## Performance Metrics

The system evaluates performance using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1-Score**: Weighted F1-score across classes
- **Confusion Matrix**: Detailed classification breakdown

## Visualization Features

### Interactive Plots

- **Query Predictions**: Time series plots with prediction labels
- **Confusion Matrix**: Heatmap visualization of classification results
- **Embedding Space**: 2D projection of high-dimensional embeddings
- **Anomaly Detection**: Scatter plots showing anomaly scores

### Export Options

- HTML interactive plots (saved to `notebooks/` directory)
- PNG/PDF static images
- CSV data exports

## Advanced Features

### Few-Shot Learning

The system implements true few-shot learning capabilities:

- Learn from minimal labeled examples (5 samples per class)
- Generalize to new classes with limited data
- Prototype-based classification for interpretability

### Anomaly Detection

Integrated anomaly detection for:

- Identifying unusual patterns in time series
- Quality control and data validation
- Outlier detection in support/query sets

### Model Comparison

Compare different architectures:

- Side-by-side performance metrics
- Visualization of embedding spaces
- Computational efficiency analysis

## Development

### Code Style

The project follows modern Python practices:

- Type hints for all function parameters and return values
- Comprehensive docstrings following Google style
- PEP8 compliance with Black formatting
- Modular design with clear separation of concerns

### Adding New Features

1. **New Architectures**: Extend `EmbeddingNetwork` class
2. **Data Generators**: Add methods to `TimeSeriesDataGenerator`
3. **Anomaly Detection**: Implement new methods in `AnomalyDetector`
4. **Visualizations**: Add new plots to `create_visualizations`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: PyTorch will automatically use CPU if CUDA is unavailable
3. **Memory Issues**: Reduce batch size or sequence length in configuration
4. **Streamlit Issues**: Check port availability and firewall settings

### Logging

Check log files in the `logs/` directory for detailed error information:

```bash
tail -f logs/timeseries_analysis.log
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{few_shot_timeseries,
  title={Few-Shot Learning for Time Series Analysis},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Few-Shot-Learning-for-Time-Series-Analysis}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework
- Plotly team for interactive visualization capabilities
- Scikit-learn team for machine learning utilities

## Future Enhancements

- [ ] Real-world dataset integration
- [ ] Advanced transformer architectures
- [ ] Multi-modal time series support
- [ ] Distributed training capabilities
- [ ] Model interpretability tools
- [ ] Automated hyperparameter optimization
- [ ] Integration with MLOps pipelines
# Few-Shot-Learning-for-Time-Series-Analysis
