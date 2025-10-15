import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import re
import traceback
import os
from extract_benchmark_results import summarize_benchmark_runs
from check_port_avaliablity import check_port_or_raise
from detailed_results_viewer import launch_detailed_viewer, create_viewer_interface
import webbrowser
import threading
from typing import List, Optional, Dict
print("Starting Gradio app code...")
# Load and process the data
local_save_directory = os.path.dirname(os.path.abspath(__file__))  
csv_filename = 'benchmark_results_summary.csv'
def generate_runs_table():
    # Use the original local path as default
    scores_sum_directory = 's3://gepeta-datasets/benchmark_results/heb_benc_results/'
    print("Summarizing benchmark runs from:", scores_sum_directory)
    # You can specify a custom local directory to save the CSV
    summarize_benchmark_runs(scores_sum_directory, local_save_directory,csv_filename)
    
def load_data():
    # Load your CSV file
    # df = pd.read_csv('benchmark_results_summary.csv')
    #  check if the file exists
    if not os.path.exists(os.path.join(local_save_directory, csv_filename)):
        print( f"CSV file not found: {os.path.join(local_save_directory, csv_filename)}",f"CSV file not found: {os.path.join(local_save_directory, csv_filename)}")
        return f"CSV file not found: {os.path.join(local_save_directory, csv_filename)}",f"CSV file not found: {os.path.join(local_save_directory, csv_filename)}"
    df = pd.read_csv(os.path.join(local_save_directory, csv_filename))

    # Clean the data - remove rows with missing model names or timestamps
    df = df.dropna(subset=['model_name', 'timestamp'])
    df = df[df['model_name'] != '']
    
    # Clean model names - remove the long path prefix
    def clean_model_name(model_name):
        if pd.isna(model_name):
            return model_name
        model_name = str(model_name)
        if model_name.startswith('/home/ec2-user/qwen-hebrew-finetuning/'):
            return model_name.replace('/home/ec2-user/qwen-hebrew-finetuning/', '')
        return model_name
    
    df['model_name'] = df['model_name'].apply(clean_model_name)
    
    # Fix timestamp format - replace hyphens with colons in time portion
    def fix_timestamp(timestamp_str):
        if pd.isna(timestamp_str):
            return timestamp_str
        # Convert to string if it's not already
        timestamp_str = str(timestamp_str)
        # Replace hyphens with colons in the time portion (after T)
        # Pattern: YYYY-MM-DDTHH-MM-SS -> YYYY-MM-DDTHH:MM:SS
        if 'T' in timestamp_str:
            date_part, time_part = timestamp_str.split('T', 1)
            time_part = time_part.replace('-', ':')
            return f"{date_part}T{time_part}"
        return timestamp_str
    
    # Apply timestamp fixing
    df['timestamp'] = df['timestamp'].apply(fix_timestamp)
    
    # Convert timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        # If parsing fails, create a dummy timestamp
        df['timestamp'] = pd.to_datetime('2024-01-01')
    
    # Remove rows where timestamp conversion failed
    df = df.dropna(subset=['timestamp'])
    
    # Add a run identifier (combination of model and timestamp)
    df['run_id'] = df['model_name'] + ' - ' + df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Get score columns (excluding std columns and metadata)
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    return df, score_columns

def get_available_runs():
    """Get list of available runs for the dropdown"""
    try:
        df, _ = load_data()
        if df.empty:
            return ["No runs available"]
        
        # Sort by timestamp (most recent first)
        df_sorted = df.sort_values('timestamp', ascending=False)
        runs = df_sorted['run_id'].tolist()
        
        # Add "Latest per model" option at the beginning
        runs.insert(0, "Latest per model")
        
        return runs
    except:
        return ["Error loading runs"]

def create_data_table():
    try:
        
        df, _ = load_data()
        # Format timestamp for display
        df_display = df.copy()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df_display
    except Exception as e:
        print(f"Error creating data table: {e}")
        #print traceback
        traceback.print_exc()

        # Return empty dataframe with error message
        return pd.DataFrame({'Error': [f'Failed to load data: {str(e)}']})

def create_benchmark_comparison(selected_runs):
    try:
        df, score_columns = load_data()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Handle run selection
        if selected_runs is None or len(selected_runs) == 0 or "Latest per model" in selected_runs:
            # Get the latest results for each model
            comparison_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        else:
            # Filter by selected runs
            comparison_df = df[df['run_id'].isin(selected_runs)]
        
        if comparison_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available for selected runs", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create scatter plot with benchmarks on x-axis
        fig = go.Figure()
        
        # Prepare benchmark names for x-axis
        benchmark_names = [col.replace('_score', '').replace('_', ' ').title() for col in score_columns]
        
        colors = px.colors.qualitative.Set1
        
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            model_name = row['model_name']
            run_time = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            # Collect scores for this run
            scores = []
            valid_benchmarks = []
            
            for j, score_col in enumerate(score_columns):
                score = row[score_col]
                if pd.notna(score):
                    scores.append(score)
                    valid_benchmarks.append(benchmark_names[j])
            
            if scores:  # Only add if there are valid scores
                fig.add_trace(go.Scatter(
                    x=valid_benchmarks,
                    y=scores,
                    mode='lines+markers',
                    name=f'{model_name} ({run_time})',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8),
                    text=[f'{score:.3f}' for score in scores],
                    textposition='top center',
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Benchmark: %{x}<br>' +
                                'Score: %{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Benchmark Scores Comparison - Selected Runs',
            xaxis_title='Benchmarks',
            yaxis_title='Scores',
            height=600,
            xaxis_tickangle=-45,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=200)  # Make room for legend
        )
        
        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        print(f"Error creating benchmark comparison: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_score_over_time():
    try:
        df, score_columns = load_data()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create subplots for different benchmarks
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['ARC AI2 Hebrew', 'HellaSwag Hebrew', 'MMLU Hebrew', 'Psychometric Scores'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Define colors for different models
        model_colors = {}
        unique_models = df['model_name'].unique()
        colors = px.colors.qualitative.Set1
        for i, model in enumerate(unique_models):
            model_colors[model] = colors[i % len(colors)]
        
        # Plot ARC AI2 Hebrew
        if 'arc_ai2_heb_score' in df.columns:
            for model in unique_models:
                model_data = df[df['model_name'] == model].sort_values('timestamp')
                if not model_data['arc_ai2_heb_score'].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['timestamp'],
                            y=model_data['arc_ai2_heb_score'],
                            mode='lines+markers',
                            name=f'{model} - ARC',
                            line=dict(color=model_colors[model]),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # Plot HellaSwag Hebrew
        if 'hellaswag_heb_score' in df.columns:
            for model in unique_models:
                model_data = df[df['model_name'] == model].sort_values('timestamp')
                if not model_data['hellaswag_heb_score'].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['timestamp'],
                            y=model_data['hellaswag_heb_score'],
                            mode='lines+markers',
                            name=f'{model} - HellaSwag',
                            line=dict(color=model_colors[model], dash='dash'),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Plot MMLU Hebrew
        if 'mmlu_heb_score' in df.columns:
            for model in unique_models:
                model_data = df[df['model_name'] == model].sort_values('timestamp')
                if not model_data['mmlu_heb_score'].isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['timestamp'],
                            y=model_data['mmlu_heb_score'],
                            mode='lines+markers',
                            name=f'{model} - MMLU',
                            line=dict(color=model_colors[model], dash='dot'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # Plot selected Psychometric scores (average)
        psychometric_cols = [col for col in score_columns if col.startswith('psychometric_heb')]
        if psychometric_cols:
            for model in unique_models:
                model_data = df[df['model_name'] == model].sort_values('timestamp')
                # Calculate average psychometric score
                psychometric_scores = model_data[psychometric_cols].mean(axis=1)
                if not psychometric_scores.isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['timestamp'],
                            y=psychometric_scores,
                            mode='lines+markers',
                            name=f'{model} - Psychometric Avg',
                            line=dict(color=model_colors[model], dash='dashdot'),
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            height=800,
            title_text="Score Trends Over Time",
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Average Score", row=2, col=2)
        
        return fig
        
    except Exception as e:
        print(f"Error creating time series: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_psychometric_detailed():
    try:
        df, score_columns = load_data()
        
        # Get psychometric columns
        psychometric_cols = [col for col in score_columns if col.startswith('psychometric_heb')]
        
        if not psychometric_cols:
            fig = go.Figure()
            fig.add_annotation(text="No psychometric data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Get latest results for each model
        latest_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        
        # Create radar chart
        fig = go.Figure()
        
        models = latest_df['model_name'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, model in enumerate(models):
            model_data = latest_df[latest_df['model_name'] == model]
            
            categories = []
            values = []
            
            for col in psychometric_cols:
                score = model_data[col].iloc[0]
                if pd.notna(score):
                    categories.append(col.replace('psychometric_heb_', '').replace('_score', '').replace('_', ' ').title())
                    values.append(score)
            
            if values:  # Only add trace if there are values
                # Close the radar chart
                categories.append(categories[0])
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model,
                    line_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Psychometric Benchmark Detailed Comparison",
            height=600
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating psychometric chart: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_model_performance_heatmap():
    try:
        df, score_columns = load_data()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Get latest results for each model
        latest_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        
        # Prepare data for heatmap
        models = latest_df['model_name'].tolist()
        
        # Create matrix
        score_matrix = []
        benchmark_names = []
        
        for score_col in score_columns:
            benchmark_name = score_col.replace('_score', '').replace('_', ' ').title()
            benchmark_names.append(benchmark_name)
            scores = latest_df[score_col].tolist()
            score_matrix.append(scores)
        
        # Transpose matrix so models are on x-axis and benchmarks on y-axis
        score_matrix = np.array(score_matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=models,
            y=benchmark_names,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=np.round(score_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title='Model Performance Heatmap',
            xaxis_title='Models',
            yaxis_title='Benchmarks',
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5)
        return fig


def get_pickle_path(row_data: Dict, dataset_column: str) -> str:
    """
    Construct pickle file path from row data
    
    Args:
        row_data: Dictionary containing row information
        dataset_column: The name of the dataset column clicked
    
    Returns:
        Path to the pickle file
    """
    print("dummy data")
    return "/home/ec2-user/test_output/hellaswag_heb/scores_sum/2025-10-15T21-51-38/details/home/ec2-user/models/Qwen3-14B/2025-10-15T21-52-23.466806/details_community|hellaswag_heb|3_2025-10-15T21-52-23.466806.parquet"
    model_name = row_data.get('model_name', '')
    timestamp = row_data.get('timestamp', '')
    
    # Convert timestamp to directory format (if needed)
    if isinstance(timestamp, str) and 'T' in timestamp:
        timestamp_dir = timestamp.replace(':', '-').split('.')[0]  # Remove microseconds
    else:
        timestamp_dir = str(timestamp)
    
    # Extract dataset name from column (e.g., 'arc_ai2_heb_score' -> 'arc_ai2_heb')
    dataset = dataset_column.replace('_score', '').replace('_std', '')
    
    # Construct path
    pickle_path = os.path.join(
        local_save_directory,
        model_name,
        timestamp_dir,
        f"{dataset}.pkl"
    )
    
    return pickle_path

def make_clickable_table():
    """Create an interactive table with clickable cells"""
    try:
        df = create_data_table()
        
        # Get score columns
        score_columns = [col for col in df.columns if col.endswith('_score')]
        
        return df, score_columns
    except Exception as e:
        print(f"Error creating clickable table: {e}")
        return pd.DataFrame({'Error': [str(e)]}), []

def handle_cell_click(row_idx: int, col_name: str, df: pd.DataFrame):
    """
    Handle cell click event and launch detailed viewer
    
    Args:
        row_idx: Index of clicked row
        col_name: Name of clicked column
        df: The dataframe
    """
    # Check if clicked column is a score column
    if not col_name.endswith('_score'):
        return None
    
    try:
        # Get row data
        row_data = df.iloc[row_idx].to_dict()
        
        # Get pickle path
        pickle_path = get_pickle_path(row_data, col_name)
        
        if not os.path.exists(pickle_path):
            print(f"Pickle file not found: {pickle_path}")
            return gr.Info(f"Data file not found: {pickle_path}")
        
        # Extract metadata
        model_name = row_data.get('model_name', 'Unknown')
        timestamp = row_data.get('timestamp', 'Unknown')
        dataset = col_name.replace('_score', '')
        
        # Launch detailed viewer in new window (requires launching as separate Gradio instance)
        viewer = create_viewer_interface(pickle_path)
        
        # Launch on different port
        import random
        port = random.randint(7700, 7799)
        
        def launch_viewer():
            viewer.launch(
                server_port=port,
                share=True,
                prevent_thread_lock=True,
                show_error=True,
                inbrowser=True
            )
        
        thread = threading.Thread(target=launch_viewer, daemon=True)
        thread.start()
        
        return gr.Info(f"Opening detailed view for {dataset} in new window on port {port}")
        
    except Exception as e:
        print(f"Error handling cell click: {e}")
        return gr.Warning(f"Error: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="Benchmark Results Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 Benchmark Results Visualization Dashboard")
    gr.Markdown("Interactive visualization of Hebrew language model benchmark results")
    
    with gr.Tabs():
        with gr.Tab("📊 Data Table"):
            gr.Markdown("### Raw Benchmark Data")
            # gr.Markdown("**💡 Tip:** Click on any score cell to view detailed results for that dataset")
            
            # Create state to store dataframe
            df_state = gr.State(value=create_data_table())
            
            data_table = gr.Dataframe(
                value=lambda: df_state.value,
                wrap=True,
                max_height=600,
                interactive=True
            )
            
            # Add click handler (Note: Gradio's Dataframe doesn't have built-in click events)
            # Instead, we'll add a selection-based approach
            
            with gr.Row():
                row_selector = gr.Number(label="Row Index", value=0, precision=0)
                col_selector = gr.Dropdown(
                    label="Dataset Column",
                    choices=[],  # Will be populated dynamically
                    value=None
                )
                view_btn = gr.Button("🔍 View Details", variant="primary")
            
            info_box = gr.Textbox(label="Status", interactive=False)
            
            def update_col_choices():
                df, score_cols = make_clickable_table()
                return gr.Dropdown(choices=score_cols)
            
            def refresh_table():
                generate_runs_table()  # Refresh data
                df = create_data_table()
                _, score_cols = make_clickable_table()
                return df, gr.Dropdown(choices=score_cols)
            
            refresh_data_btn = gr.Button("🔄 Refresh Data", variant="secondary")
            
            # Handle view button click
            def on_view_click(row_idx, col_name, df):
                if col_name is None:
                    return "Please select a dataset column"
                try:
                    result = handle_cell_click(int(row_idx), col_name, df)
                    return f"Launched viewer for row {row_idx}, column {col_name}"
                except Exception as e:
                    return f"Error: {str(e)}"
            
            view_btn.click(
                fn=on_view_click,
                inputs=[row_selector, col_selector, df_state],
                outputs=info_box
            )
            
            refresh_data_btn.click(
                fn=refresh_table,
                outputs=[data_table, col_selector]
            )
            
            # Initialize column choices on load
            demo.load(fn=update_col_choices, outputs=col_selector)        
        with gr.Tab("📈 Benchmark Comparison"):
            gr.Markdown("### Compare Scores Across Benchmarks")
            gr.Markdown("Select specific runs to compare, or use 'Latest per model' to compare the most recent results for each model.")
            
            # Run selection dropdown
            run_selector = gr.Dropdown(
                choices=get_available_runs(),
                value=["Latest per model"],
                multiselect=True,
                label="Select Runs to Compare",
                info="Choose which runs to display in the comparison chart"
            )
            
            comparison_plot = gr.Plot(value=create_benchmark_comparison(["Latest per model"]))
            
            # Update chart when selection changes
            run_selector.change(
                fn=create_benchmark_comparison,
                inputs=[run_selector],
                outputs=comparison_plot
            )
            
            with gr.Row():
                refresh_comparison_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
                refresh_runs_btn = gr.Button("🔄 Refresh Run List", variant="secondary")
            
            refresh_comparison_btn.click(
                fn=create_benchmark_comparison,
                inputs=[run_selector],
                outputs=comparison_plot
            )
            refresh_runs_btn.click(
                fn=get_available_runs,
                outputs=run_selector
            )
        
        with gr.Tab("⏰ Scores Over Time"):
            gr.Markdown("### Performance Trends Over Time")
            time_plot = gr.Plot(value=create_score_over_time())
            
            refresh_time_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_time_btn.click(fn=create_score_over_time, outputs=time_plot)
        
        with gr.Tab("🎯 Psychometric Details"):
            gr.Markdown("### Detailed Psychometric Benchmark Analysis")
            psychometric_plot = gr.Plot(value=create_psychometric_detailed())
            
            refresh_psychometric_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_psychometric_btn.click(fn=create_psychometric_detailed, outputs=psychometric_plot)
        
        with gr.Tab("🔥 Performance Heatmap"):
            gr.Markdown("### Model Performance Heatmap")
            heatmap_plot = gr.Plot(value=create_model_performance_heatmap())
            
            refresh_heatmap_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_heatmap_btn.click(fn=create_model_performance_heatmap, outputs=heatmap_plot)
PORT = 7680
# Launch the app
if __name__ == "__main__":
    print("Starting Gradio app...")
    # generate_runs_table()  # Initial data generation
    check_port_or_raise(PORT, timeout=3,auto_kill=True, retry=True)
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        debug=False,
        prevent_thread_lock=False
    )

# kill previous process on port 7680
# lsof -ti:7680 | xargs kill -9 2>/dev/null || echo "No process found on port 7680"

# check if the protocol is avaliable
# netstat -tlnp | grep 7680
# start server in background
# sudo systemctl daemon-reload
# sudo systemctl enable gradio
# sudo systemctl start gradio
## check status
# sudo systemctl status gradio
## monitor logs
# sudo journalctl -u gradio -f

# nohup python scores_dashboard.py > gradio.log 2>&1 &
# import gradio as gr
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# from datetime import datetime
# import re

# # Load and process the data
# def load_data():
#     # Load your CSV file
#     # df = pd.read_csv('benchmark_results_summary.csv')
#     df = pd.read_csv('/home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum/benchmark_results_summary.csv')

#     # Clean the data - remove rows with missing model names or timestamps
#     df = df.dropna(subset=['model_name', 'timestamp'])
#     df = df[df['model_name'] != '']
    
#     # Fix timestamp format - replace hyphens with colons in time portion
#     def fix_timestamp(timestamp_str):
#         if pd.isna(timestamp_str):
#             return timestamp_str
#         # Convert to string if it's not already
#         timestamp_str = str(timestamp_str)
#         # Replace hyphens with colons in the time portion (after T)
#         # Pattern: YYYY-MM-DDTHH-MM-SS -> YYYY-MM-DDTHH:MM:SS
#         if 'T' in timestamp_str:
#             date_part, time_part = timestamp_str.split('T', 1)
#             time_part = time_part.replace('-', ':')
#             return f"{date_part}T{time_part}"
#         return timestamp_str
    
#     # Apply timestamp fixing
#     df['timestamp'] = df['timestamp'].apply(fix_timestamp)
    
#     # Convert timestamp to datetime
#     try:
#         df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#     except Exception as e:
#         print(f"Error parsing timestamps: {e}")
#         # If parsing fails, create a dummy timestamp
#         df['timestamp'] = pd.to_datetime('2024-01-01')
    
#     # Remove rows where timestamp conversion failed
#     df = df.dropna(subset=['timestamp'])
    
#     # Get score columns (excluding std columns and metadata)
#     score_columns = [col for col in df.columns if col.endswith('_score')]
    
#     return df, score_columns

# def create_data_table():
#     try:
#         df, _ = load_data()
#         # Format timestamp for display
#         df_display = df.copy()
#         df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
#         return df_display
#     except Exception as e:
#         print(f"Error creating data table: {e}")
#         # Return empty dataframe with error message
#         return pd.DataFrame({'Error': [f'Failed to load data: {str(e)}']})

# def create_benchmark_comparison():
#     try:
#         df, score_columns = load_data()
        
#         if df.empty:
#             fig = go.Figure()
#             fig.add_annotation(text="No data available", 
#                              xref="paper", yref="paper", x=0.5, y=0.5)
#             return fig
        
#         # Get the latest results for each model
#         latest_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        
#         # Prepare data for plotting
#         models = latest_df['model_name'].tolist()
        
#         # Create a grouped bar chart
#         fig = go.Figure()
        
#         colors = px.colors.qualitative.Set3
        
#         for i, score_col in enumerate(score_columns):
#             # Clean column name for display
#             benchmark_name = score_col.replace('_score', '').replace('_', ' ').title()
            
#             scores = latest_df[score_col].tolist()
            
#             fig.add_trace(go.Bar(
#                 name=benchmark_name,
#                 x=models,
#                 y=scores,
#                 marker_color=colors[i % len(colors)],
#                 text=[f'{score:.2f}' if pd.notna(score) else 'N/A' for score in scores],
#                 textposition='auto',
#             ))
        
#         fig.update_layout(
#             title='Benchmark Scores Comparison Across Models',
#             xaxis_title='Models',
#             yaxis_title='Scores',
#             barmode='group',
#             height=600,
#             xaxis_tickangle=-45,
#             legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=1.02,
#                 xanchor="right",
#                 x=1
#             )
#         )
        
#         return fig
        
#     except Exception as e:
#         print(f"Error creating benchmark comparison: {e}")
#         fig = go.Figure()
#         fig.add_annotation(text=f"Error creating chart: {str(e)}", 
#                          xref="paper", yref="paper", x=0.5, y=0.5)
#         return fig

# def create_score_over_time():
#     try:
#         df, score_columns = load_data()
        
#         if df.empty:
#             fig = go.Figure()
#             fig.add_annotation(text="No data available", 
#                              xref="paper", yref="paper", x=0.5, y=0.5)
#             return fig
        
#         # Create subplots for different benchmarks
#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=['ARC AI2 Hebrew', 'HellaSwag Hebrew', 'MMLU Hebrew', 'Psychometric Scores'],
#             specs=[[{"secondary_y": False}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}]]
#         )
        
#         # Define colors for different models
#         model_colors = {}
#         unique_models = df['model_name'].unique()
#         colors = px.colors.qualitative.Set1
#         for i, model in enumerate(unique_models):
#             model_colors[model] = colors[i % len(colors)]
        
#         # Plot ARC AI2 Hebrew
#         if 'arc_ai2_heb_score' in df.columns:
#             for model in unique_models:
#                 model_data = df[df['model_name'] == model].sort_values('timestamp')
#                 if not model_data['arc_ai2_heb_score'].isna().all():
#                     fig.add_trace(
#                         go.Scatter(
#                             x=model_data['timestamp'],
#                             y=model_data['arc_ai2_heb_score'],
#                             mode='lines+markers',
#                             name=f'{model} - ARC',
#                             line=dict(color=model_colors[model]),
#                             showlegend=True
#                         ),
#                         row=1, col=1
#                     )
        
#         # Plot HellaSwag Hebrew
#         if 'hellaswag_heb_score' in df.columns:
#             for model in unique_models:
#                 model_data = df[df['model_name'] == model].sort_values('timestamp')
#                 if not model_data['hellaswag_heb_score'].isna().all():
#                     fig.add_trace(
#                         go.Scatter(
#                             x=model_data['timestamp'],
#                             y=model_data['hellaswag_heb_score'],
#                             mode='lines+markers',
#                             name=f'{model} - HellaSwag',
#                             line=dict(color=model_colors[model], dash='dash'),
#                             showlegend=False
#                         ),
#                         row=1, col=2
#                     )
        
#         # Plot MMLU Hebrew
#         if 'mmlu_heb_score' in df.columns:
#             for model in unique_models:
#                 model_data = df[df['model_name'] == model].sort_values('timestamp')
#                 if not model_data['mmlu_heb_score'].isna().all():
#                     fig.add_trace(
#                         go.Scatter(
#                             x=model_data['timestamp'],
#                             y=model_data['mmlu_heb_score'],
#                             mode='lines+markers',
#                             name=f'{model} - MMLU',
#                             line=dict(color=model_colors[model], dash='dot'),
#                             showlegend=False
#                         ),
#                         row=2, col=1
#                     )
        
#         # Plot selected Psychometric scores (average)
#         psychometric_cols = [col for col in score_columns if col.startswith('psychometric_heb')]
#         if psychometric_cols:
#             for model in unique_models:
#                 model_data = df[df['model_name'] == model].sort_values('timestamp')
#                 # Calculate average psychometric score
#                 psychometric_scores = model_data[psychometric_cols].mean(axis=1)
#                 if not psychometric_scores.isna().all():
#                     fig.add_trace(
#                         go.Scatter(
#                             x=model_data['timestamp'],
#                             y=psychometric_scores,
#                             mode='lines+markers',
#                             name=f'{model} - Psychometric Avg',
#                             line=dict(color=model_colors[model], dash='dashdot'),
#                             showlegend=False
#                         ),
#                         row=2, col=2
#                     )
        
#         fig.update_layout(
#             height=800,
#             title_text="Score Trends Over Time",
#             showlegend=True
#         )
        
#         # Update y-axis labels
#         fig.update_yaxes(title_text="Score", row=1, col=1)
#         fig.update_yaxes(title_text="Score", row=1, col=2)
#         fig.update_yaxes(title_text="Score", row=2, col=1)
#         fig.update_yaxes(title_text="Average Score", row=2, col=2)
        
#         return fig
        
#     except Exception as e:
#         print(f"Error creating time series: {e}")
#         fig = go.Figure()
#         fig.add_annotation(text=f"Error creating chart: {str(e)}", 
#                          xref="paper", yref="paper", x=0.5, y=0.5)
#         return fig

# def create_psychometric_detailed():
#     try:
#         df, score_columns = load_data()
        
#         # Get psychometric columns
#         psychometric_cols = [col for col in score_columns if col.startswith('psychometric_heb')]
        
#         if not psychometric_cols:
#             fig = go.Figure()
#             fig.add_annotation(text="No psychometric data available", 
#                              xref="paper", yref="paper", x=0.5, y=0.5)
#             return fig
        
#         # Get latest results for each model
#         latest_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        
#         # Create radar chart
#         fig = go.Figure()
        
#         models = latest_df['model_name'].unique()
#         colors = px.colors.qualitative.Set1
        
#         for i, model in enumerate(models):
#             model_data = latest_df[latest_df['model_name'] == model]
            
#             categories = []
#             values = []
            
#             for col in psychometric_cols:
#                 score = model_data[col].iloc[0]
#                 if pd.notna(score):
#                     categories.append(col.replace('psychometric_heb_', '').replace('_score', '').replace('_', ' ').title())
#                     values.append(score)
            
#             if values:  # Only add trace if there are values
#                 # Close the radar chart
#                 categories.append(categories[0])
#                 values.append(values[0])
                
#                 fig.add_trace(go.Scatterpolar(
#                     r=values,
#                     theta=categories,
#                     fill='toself',
#                     name=model.split('/')[-1] if '/' in model else model,  # Shorten model name
#                     line_color=colors[i % len(colors)]
#                 ))
        
#         fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[0, 1]
#                 )),
#             showlegend=True,
#             title="Psychometric Benchmark Detailed Comparison",
#             height=600
#         )
        
#         return fig
        
#     except Exception as e:
#         print(f"Error creating psychometric chart: {e}")
#         fig = go.Figure()
#         fig.add_annotation(text=f"Error creating chart: {str(e)}", 
#                          xref="paper", yref="paper", x=0.5, y=0.5)
#         return fig

# def create_model_performance_heatmap():
#     try:
#         df, score_columns = load_data()
        
#         if df.empty:
#             fig = go.Figure()
#             fig.add_annotation(text="No data available", 
#                              xref="paper", yref="paper", x=0.5, y=0.5)
#             return fig
        
#         # Get latest results for each model
#         latest_df = df.loc[df.groupby('model_name')['timestamp'].idxmax()]
        
#         # Prepare data for heatmap
#         models = latest_df['model_name'].tolist()
#         model_names = [model.split('/')[-1] if '/' in model else model for model in models]  # Shorten names
        
#         # Create matrix
#         score_matrix = []
#         benchmark_names = []
        
#         for score_col in score_columns:
#             benchmark_name = score_col.replace('_score', '').replace('_', ' ').title()
#             benchmark_names.append(benchmark_name)
#             scores = latest_df[score_col].tolist()
#             score_matrix.append(scores)
        
#         # Transpose matrix so models are on x-axis and benchmarks on y-axis
#         score_matrix = np.array(score_matrix)
        
#         fig = go.Figure(data=go.Heatmap(
#             z=score_matrix,
#             x=model_names,
#             y=benchmark_names,
#             colorscale='RdYlGn',
#             zmin=0,
#             zmax=1,
#             text=np.round(score_matrix, 3),
#             texttemplate="%{text}",
#             textfont={"size": 10},
#             colorbar=dict(title="Score")
#         ))
        
#         fig.update_layout(
#             title='Model Performance Heatmap',
#             xaxis_title='Models',
#             yaxis_title='Benchmarks',
#             height=600,
#             xaxis_tickangle=-45
#         )
        
#         return fig
        
#     except Exception as e:
#         print(f"Error creating heatmap: {e}")
#         fig = go.Figure()
#         fig.add_annotation(text=f"Error creating chart: {str(e)}", 
#                          xref="paper", yref="paper", x=0.5, y=0.5)
#         return fig

# # Create Gradio interface
# with gr.Blocks(title="Benchmark Results Visualization", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# 🏆 Benchmark Results Visualization Dashboard")
#     gr.Markdown("Interactive visualization of Hebrew language model benchmark results")
    
#     with gr.Tabs():
#         with gr.Tab("📊 Data Table"):
#             gr.Markdown("### Raw Benchmark Data")
#             data_table = gr.Dataframe(
#                 value=create_data_table(),
#                 wrap=True,
#                 max_height=600,
#                 interactive=False
#             )
            
#             refresh_data_btn = gr.Button("🔄 Refresh Data", variant="secondary")
#             refresh_data_btn.click(fn=create_data_table, outputs=data_table)
        
#         with gr.Tab("📈 Benchmark Comparison"):
#             gr.Markdown("### Compare Latest Scores Across All Benchmarks")
#             comparison_plot = gr.Plot(value=create_benchmark_comparison())
            
#             refresh_comparison_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
#             refresh_comparison_btn.click(fn=create_benchmark_comparison, outputs=comparison_plot)
        
#         with gr.Tab("⏰ Scores Over Time"):
#             gr.Markdown("### Performance Trends Over Time")
#             time_plot = gr.Plot(value=create_score_over_time())
            
#             refresh_time_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
#             refresh_time_btn.click(fn=create_score_over_time, outputs=time_plot)
        
#         with gr.Tab("🎯 Psychometric Details"):
#             gr.Markdown("### Detailed Psychometric Benchmark Analysis")
#             psychometric_plot = gr.Plot(value=create_psychometric_detailed())
            
#             refresh_psychometric_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
#             refresh_psychometric_btn.click(fn=create_psychometric_detailed, outputs=psychometric_plot)
        
#         with gr.Tab("🔥 Performance Heatmap"):
#             gr.Markdown("### Model Performance Heatmap")
#             heatmap_plot = gr.Plot(value=create_model_performance_heatmap())
            
#             refresh_heatmap_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
#             refresh_heatmap_btn.click(fn=create_model_performance_heatmap, outputs=heatmap_plot)

# # Launch the app
# if __name__ == "__main__":
#     demo.launch(
#         share=True,
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True
#     )
