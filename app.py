"""
Streamlit Web UI for Model Pruning Analysis

This module provides an interactive web interface for running pruning experiments,
visualizing results, and comparing different pruning strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from datetime import datetime
import logging

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from model_pruning import ModelPruner, ModernMLP
from config import ConfigManager, ExperimentConfig
from database import MockDatabase, ExperimentManager
from visualization import PruningVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Model Pruning Analysis",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'experiment_manager' not in st.session_state:
    st.session_state.experiment_manager = ExperimentManager()
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">✂️ Model Pruning Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Experiment selection
        st.subheader("Experiment Management")
        
        # Load existing experiments
        experiments_df = st.session_state.experiment_manager.get_experiment_comparison()
        
        if not experiments_df.empty:
            st.write("**Existing Experiments:**")
            for idx, row in experiments_df.iterrows():
                with st.expander(f"Experiment {row['experiment_id']}"):
                    st.write(f"**Method:** {row['pruning_method']}")
                    st.write(f"**Amount:** {row['pruning_amount']:.1%}")
                    st.write(f"**Accuracy:** {row['final_accuracy']:.2f}%")
                    st.write(f"**Size Reduction:** {row['model_size_reduction']:.1%}")
        
        # New experiment configuration
        st.subheader("New Experiment")
        
        with st.form("experiment_config"):
            experiment_name = st.text_input("Experiment Name", value=f"exp_{int(time.time())}")
            
            # Model configuration
            st.write("**Model Configuration:**")
            hidden_layers = st.multiselect(
                "Hidden Layer Sizes", 
                options=[100, 200, 300, 400, 500],
                default=[300, 100]
            )
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            
            # Training configuration
            st.write("**Training Configuration:**")
            epochs = st.slider("Training Epochs", 1, 20, 5)
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
            
            # Pruning configuration
            st.write("**Pruning Configuration:**")
            pruning_method = st.selectbox(
                "Pruning Method",
                ["l1_unstructured", "l2_unstructured", "random_unstructured", "structured"]
            )
            pruning_amount = st.slider("Pruning Amount", 0.1, 0.9, 0.5, 0.05)
            retrain_after_pruning = st.checkbox("Retrain After Pruning", value=True)
            
            if retrain_after_pruning:
                retrain_epochs = st.slider("Retrain Epochs", 1, 10, 3)
            else:
                retrain_epochs = 0
            
            # Device selection
            device_options = ["auto", "cpu"]
            if torch.cuda.is_available():
                device_options.append("cuda")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_options.append("mps")
            
            device = st.selectbox("Device", device_options)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Run Experiment", use_container_width=True)
            
            if submitted:
                if not hidden_layers:
                    st.error("Please select at least one hidden layer size!")
                else:
                    # Create configuration
                    config = ExperimentConfig(
                        model={
                            'hidden_sizes': hidden_layers,
                            'dropout_rate': dropout_rate
                        },
                        training={
                            'epochs': epochs,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        },
                        pruning={
                            'pruning_method': pruning_method,
                            'pruning_amount': pruning_amount,
                            'retrain_after_pruning': retrain_after_pruning,
                            'retrain_epochs': retrain_epochs
                        },
                        data={},
                        device=device
                    )
                    
                    # Run experiment
                    run_experiment(experiment_name, config)


def run_experiment(experiment_name: str, config: ExperimentConfig):
    """Run a pruning experiment"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pruner
        status_text.text("Initializing model pruner...")
        progress_bar.progress(10)
        
        pruner = ModelPruner(config)
        
        # Load data
        status_text.text("Loading MNIST dataset...")
        progress_bar.progress(20)
        
        train_loader, test_loader = pruner.load_data()
        
        # Create model
        status_text.text("Creating model...")
        progress_bar.progress(30)
        
        model = ModernMLP(
            hidden_sizes=config.model.hidden_sizes,
            dropout_rate=config.model.dropout_rate
        ).to(pruner.device)
        
        # Train baseline model
        status_text.text("Training baseline model...")
        progress_bar.progress(40)
        
        model = pruner.train_model(model, train_loader, test_loader)
        baseline_accuracy = pruner.metrics.get_latest('test_accuracy')
        
        # Apply pruning
        status_text.text("Applying pruning...")
        progress_bar.progress(60)
        
        model = pruner.prune_model(model, config.pruning.pruning_amount)
        
        # Remove reparameterization
        status_text.text("Finalizing pruning...")
        progress_bar.progress(70)
        
        model = pruner.remove_pruning_reparameterization(model)
        
        # Evaluate pruned model
        status_text.text("Evaluating pruned model...")
        progress_bar.progress(80)
        
        pruned_accuracy = pruner.evaluate(model, test_loader)
        
        # Retrain if configured
        if config.pruning.retrain_after_pruning:
            status_text.text("Retraining pruned model...")
            progress_bar.progress(90)
            
            model = pruner.train_model(model, train_loader, test_loader)
            final_accuracy = pruner.metrics.get_latest('test_accuracy')
        else:
            final_accuracy = pruned_accuracy
        
        # Compile results
        status_text.text("Compiling results...")
        progress_bar.progress(95)
        
        initial_size = model.get_model_size()
        pruned_size = model.get_model_size()
        
        results = {
            'baseline_accuracy': baseline_accuracy,
            'pruned_accuracy': pruned_accuracy,
            'final_accuracy': final_accuracy,
            'accuracy_drop': baseline_accuracy - final_accuracy,
            'model_size_reduction': (initial_size['total_parameters'] - pruned_size['total_parameters']) / initial_size['total_parameters'],
            'initial_model_size': initial_size,
            'pruned_model_size': pruned_size,
            'config': config.__dict__,
            'metrics': pruner.metrics.to_dict()
        }
        
        # Save experiment
        st.session_state.experiment_manager.create_experiment(experiment_name, config.__dict__)
        st.session_state.experiment_manager.update_experiment_results(experiment_name, results)
        
        # Store results in session state
        st.session_state.current_experiment = experiment_name
        st.session_state.experiment_results = results
        
        progress_bar.progress(100)
        status_text.text("✅ Experiment completed successfully!")
        
        # Show success message
        st.success(f"🎉 Experiment '{experiment_name}' completed!")
        
        # Auto-refresh the page to show results
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Experiment failed: {str(e)}")
        logger.error(f"Experiment failed: {e}")


def display_results():
    """Display experiment results"""
    
    if st.session_state.experiment_results is None:
        st.info("👈 Run an experiment from the sidebar to see results here!")
        return
    
    results = st.session_state.experiment_results
    
    # Results header
    st.header("📊 Experiment Results")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Accuracy", 
            f"{results['baseline_accuracy']:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Final Accuracy", 
            f"{results['final_accuracy']:.2f}%",
            delta=f"{results['accuracy_drop']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Model Size Reduction", 
            f"{results['model_size_reduction']:.1%}",
            delta=None
        )
    
    with col4:
        pruning_ratio = results.get('pruning_ratio', 0.5)
        st.metric(
            "Pruning Ratio", 
            f"{pruning_ratio:.1%}",
            delta=None
        )
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔍 Model Analysis", "📊 Visualizations", "📋 Details"])
    
    with tab1:
        display_performance_tab(results)
    
    with tab2:
        display_model_analysis_tab(results)
    
    with tab3:
        display_visualizations_tab(results)
    
    with tab4:
        display_details_tab(results)


def display_performance_tab(results):
    """Display performance analysis tab"""
    
    st.subheader("Performance Analysis")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=['Accuracy'],
            y=[results['baseline_accuracy']],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Pruned',
            x=['Accuracy'],
            y=[results['final_accuracy']],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance vs Size trade-off
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[100],
            y=[results['baseline_accuracy']],
            mode='markers+text',
            text=['Baseline'],
            textposition='top center',
            marker=dict(size=15, color='blue'),
            name='Baseline Model'
        ))
        
        fig.add_trace(go.Scatter(
            x=[100 - results['model_size_reduction'] * 100],
            y=[results['final_accuracy']],
            mode='markers+text',
            text=['Pruned'],
            textposition='top center',
            marker=dict(size=15, color='red'),
            name='Pruned Model'
        ))
        
        fig.update_layout(
            title="Performance vs Size Trade-off",
            xaxis_title="Model Size (%)",
            yaxis_title="Accuracy (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Training metrics if available
    if 'metrics' in results and results['metrics']:
        st.subheader("Training Progress")
        
        metrics = results['metrics']
        
        if 'train_loss' in metrics and 'test_accuracy' in metrics:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Loss', 'Test Accuracy')
            )
            
            fig.add_trace(
                go.Scatter(y=metrics['train_loss'], mode='lines+markers', name='Loss'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(y=metrics['test_accuracy'], mode='lines+markers', name='Accuracy'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def display_model_analysis_tab(results):
    """Display model analysis tab"""
    
    st.subheader("Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model size comparison
        initial_size = results['initial_model_size']
        pruned_size = results['pruned_model_size']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Initial',
            x=['Parameters', 'Size (MB)'],
            y=[initial_size['total_parameters'], initial_size['model_size_mb']],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Pruned',
            x=['Parameters', 'Size (MB)'],
            y=[pruned_size['total_parameters'], pruned_size['model_size_mb']],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Model Size Comparison",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pruning effectiveness
        accuracy_drop = results['accuracy_drop']
        size_reduction = results['model_size_reduction']
        
        # Calculate efficiency score (size reduction / accuracy drop)
        efficiency_score = size_reduction / max(accuracy_drop / 100, 0.001)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=efficiency_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Pruning Efficiency Score"},
            delta={'reference': 1.0},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 3], 'color': "yellow"},
                    {'range': [3, 5], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model statistics table
    st.subheader("Model Statistics")
    
    stats_data = {
        'Metric': [
            'Total Parameters (Initial)',
            'Total Parameters (Pruned)',
            'Parameter Reduction',
            'Model Size (Initial)',
            'Model Size (Pruned)',
            'Size Reduction'
        ],
        'Value': [
            f"{initial_size['total_parameters']:,}",
            f"{pruned_size['total_parameters']:,}",
            f"{results['model_size_reduction']:.1%}",
            f"{initial_size['model_size_mb']:.2f} MB",
            f"{pruned_size['model_size_mb']:.2f} MB",
            f"{results['model_size_reduction']:.1%}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)


def display_visualizations_tab(results):
    """Display visualizations tab"""
    
    st.subheader("Visualizations")
    
    # Create visualizer
    visualizer = PruningVisualizer()
    
    # Generate visualizations
    if st.button("🎨 Generate Visualizations"):
        with st.spinner("Generating visualizations..."):
            try:
                # This would generate actual plots - for demo, we'll show placeholders
                st.info("Visualization generation would happen here in a full implementation")
                
                # Placeholder charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weight Distribution")
                    # Placeholder for weight distribution plot
                    st.info("Weight distribution plot would be displayed here")
                
                with col2:
                    st.subheader("Layer-wise Analysis")
                    # Placeholder for layer-wise analysis
                    st.info("Layer-wise pruning analysis would be displayed here")
                
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")


def display_details_tab(results):
    """Display detailed information tab"""
    
    st.subheader("Experiment Details")
    
    # Configuration details
    st.write("**Configuration:**")
    config = results.get('config', {})
    
    if config:
        config_df = pd.DataFrame([
            {'Parameter': 'Pruning Method', 'Value': config.get('pruning_method', 'N/A')},
            {'Parameter': 'Pruning Amount', 'Value': f"{config.get('pruning_amount', 0):.1%}"},
            {'Parameter': 'Learning Rate', 'Value': config.get('learning_rate', 'N/A')},
            {'Parameter': 'Batch Size', 'Value': config.get('batch_size', 'N/A')},
            {'Parameter': 'Epochs', 'Value': config.get('epochs', 'N/A')},
            {'Parameter': 'Device', 'Value': config.get('device', 'N/A')}
        ])
        
        st.dataframe(config_df, use_container_width=True)
    
    # Raw results JSON
    st.subheader("Raw Results")
    
    with st.expander("View Raw JSON Data"):
        st.json(results)


def display_experiment_comparison():
    """Display comparison of all experiments"""
    
    st.header("📊 Experiment Comparison")
    
    experiments_df = st.session_state.experiment_manager.get_experiment_comparison()
    
    if experiments_df.empty:
        st.info("No experiments to compare. Run some experiments first!")
        return
    
    # Summary table
    st.subheader("Summary Table")
    st.dataframe(experiments_df, use_container_width=True)
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            experiments_df, 
            x='pruning_method', 
            y='final_accuracy',
            title='Final Accuracy by Pruning Method',
            color='final_accuracy',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Size reduction comparison
        fig = px.bar(
            experiments_df, 
            x='pruning_method', 
            y='model_size_reduction',
            title='Model Size Reduction by Pruning Method',
            color='model_size_reduction',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Accuracy vs Size Reduction
    fig = px.scatter(
        experiments_df,
        x='model_size_reduction',
        y='final_accuracy',
        color='pruning_method',
        size='total_parameters',
        hover_data=['experiment_id', 'accuracy_drop'],
        title='Performance vs Size Reduction Trade-off'
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Main app layout
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Results", "📈 Comparison"])
    
    with tab1:
        main()
    
    with tab2:
        display_results()
    
    with tab3:
        display_experiment_comparison()
