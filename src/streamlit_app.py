"""
Streamlit Interface for Few-Shot Time Series Learning

This module provides an interactive web interface for exploring and analyzing
time series data using few-shot learning techniques.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path

from src.timeseries_analyzer import TimeSeriesAnalyzer, TimeSeriesDataGenerator


def load_config() -> dict:
    """Load configuration from YAML file."""
    try:
        with open("config/config.yaml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config/config.yaml exists.")
        return {}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Few-Shot Time Series Learning",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Few-Shot Time Series Learning Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Load configuration
    config = load_config()
    if not config:
        st.stop()
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    sequence_length = st.sidebar.slider(
        "Sequence Length", 
        min_value=20, 
        max_value=100, 
        value=config.get("data", {}).get("sequence_length", 50)
    )
    
    num_support_samples = st.sidebar.slider(
        "Support Samples per Class", 
        min_value=2, 
        max_value=10, 
        value=config.get("data", {}).get("num_support_samples", 5)
    )
    
    num_query_samples = st.sidebar.slider(
        "Query Samples per Class", 
        min_value=5, 
        max_value=20, 
        value=config.get("data", {}).get("num_query_samples", 10)
    )
    
    noise_level = st.sidebar.slider(
        "Noise Level", 
        min_value=0.0, 
        max_value=0.5, 
        value=config.get("data", {}).get("noise_level", 0.1),
        step=0.01
    )
    
    num_classes = st.sidebar.selectbox(
        "Number of Classes", 
        options=[2, 3, 4], 
        index=0
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    architecture = st.sidebar.selectbox(
        "Embedding Architecture",
        options=["conv", "lstm", "gru", "transformer"],
        index=0
    )
    
    embedding_dim = st.sidebar.slider(
        "Embedding Dimension",
        min_value=16,
        max_value=128,
        value=config.get("model", {}).get("embedding_dim", 32)
    )
    
    dropout = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=config.get("model", {}).get("dropout", 0.1),
        step=0.01
    )
    
    # Update config with user selections
    config["data"]["sequence_length"] = sequence_length
    config["data"]["num_support_samples"] = num_support_samples
    config["data"]["num_query_samples"] = num_query_samples
    config["data"]["noise_level"] = noise_level
    config["data"]["num_classes"] = num_classes
    config["model"]["embedding_dim"] = embedding_dim
    config["model"]["dropout"] = dropout
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Analysis Results")
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Running analysis..."):
                try:
                    # Initialize analyzer with updated config
                    analyzer = TimeSeriesAnalyzer()
                    analyzer.config = config
                    analyzer.data_generator = TimeSeriesDataGenerator(config)
                    
                    # Run analysis
                    results = analyzer.run_analysis()
                    
                    # Store results in session state
                    st.session_state.results = results
                    
                    st.success("Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    with col2:
        st.header("Quick Stats")
        if "results" in st.session_state:
            results = st.session_state.results
            
            # Display metrics
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
            st.metric("Precision", f"{results['precision']:.2%}")
            st.metric("Recall", f"{results['recall']:.2%}")
            st.metric("F1-Score", f"{results['f1_score']:.2%}")
    
    # Display results if available
    if "results" in st.session_state:
        results = st.session_state.results
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Predictions", 
            "🎯 Confusion Matrix", 
            "🔍 Embeddings", 
            "⚠️ Anomalies"
        ])
        
        with tab1:
            st.subheader("Query Predictions")
            
            # Plot query predictions
            fig = go.Figure()
            
            query_data = results["query_data"]
            predictions = results["predictions"]
            true_labels = results["true_labels"]
            
            colors = px.colors.qualitative.Set1
            
            for i in range(min(10, len(query_data))):
                color = colors[predictions[i] % len(colors)]
                fig.add_trace(go.Scatter(
                    y=query_data[i],
                    mode="lines",
                    name=f"Query {i+1} (Pred: {predictions[i]}, True: {true_labels[i]})",
                    line=dict(color=color, width=2),
                    opacity=0.8
                ))
            
            fig.update_layout(
                title="Query Time Series Predictions",
                xaxis_title="Time Steps",
                yaxis_title="Value",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(results["true_labels"], results["predictions"])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Class",
                yaxis_title="True Class",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            from sklearn.metrics import classification_report
            report = classification_report(
                results["true_labels"], 
                results["predictions"], 
                output_dict=True
            )
            
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        with tab3:
            st.subheader("Embedding Space Visualization")
            
            # Simple 2D projection of distances
            distances = results["distances"]
            
            fig = go.Figure()
            
            for class_idx in range(num_classes):
                class_mask = results["true_labels"] == class_idx
                class_distances = distances[class_mask]
                
                fig.add_trace(go.Scatter(
                    x=class_distances[:, 0],
                    y=class_distances[:, 1],
                    mode="markers",
                    name=f"Class {class_idx}",
                    marker=dict(size=10, opacity=0.7)
                ))
            
            fig.update_layout(
                title="Embedding Space (Distance-based Projection)",
                xaxis_title="Distance to Prototype 0",
                yaxis_title="Distance to Prototype 1",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Anomaly Detection Results")
            
            anomaly_scores = results["anomaly_scores"]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(anomaly_scores))),
                y=anomaly_scores,
                mode="markers",
                marker=dict(
                    color=anomaly_scores,
                    colorscale="RdYlBu_r",
                    size=8,
                    colorbar=dict(title="Anomaly Score")
                ),
                name="Anomaly Scores"
            ))
            
            fig.update_layout(
                title="Anomaly Detection Results",
                xaxis_title="Sample Index",
                yaxis_title="Anomaly Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            num_anomalies = np.sum(anomaly_scores == -1)
            st.metric("Number of Anomalies Detected", num_anomalies)
            st.metric("Anomaly Rate", f"{num_anomalies/len(anomaly_scores):.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Few-Shot Time Series Learning** - A comprehensive analysis tool for "
        "time series classification using prototypical networks and anomaly detection."
    )


if __name__ == "__main__":
    main()
