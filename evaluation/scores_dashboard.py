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
from s3_utils import download_s3_file
from extract_benchmark_results import summarize_benchmark_runs
from check_port_avaliablity import check_port_or_raise
from detailed_results_viewer import create_detailed_viewer   
import webbrowser
import threading
from typing import List, Optional, Dict

print("Starting Gradio app code...")
# Load and process the data
local_save_directory = os.path.dirname(os.path.abspath(__file__))  
csv_filename = 'benchmark_results_summary.csv'
BUCKET_NAME = 'gepeta-datasets'
scores_sum_directory = f's3://{BUCKET_NAME}/benchmark_results/heb_benc_results/'
def generate_runs_table():
    # Use the original local path as default
    
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
        if model_name.startswith('/home/ec2-user/models/'):
            return model_name.replace('/home/ec2-user/models/', '')
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
    df = df.reset_index(drop=True)
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
def create_benchmark_comparison(selected_runs, plot_mode="Lines + Markers"):
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
        
        # Determine plot mode
        mode = 'lines+markers' if plot_mode == "Lines + Markers" else 'markers'
        
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            model_name = row['model_name']
            run_time = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            heb_benchmarks = ["psychometric_heb_understanding_hebrew_score",
                              "psychometric_heb_sentence_text_hebrew_score",
                              "psychometric_heb_sentence_complete_hebrew_score",
                              "psychometric_heb_analogies_hebrew_score"]
            eng_benchmarks = ["psychometric_heb_restatement_english_score",
                              "psychometric_heb_sentence_text_english_score",
                              "psychometric_heb_sentence_complete_english_score"]
            GATHER_PSYCHOMETRIC_TOPICS = True
            #change score_columns to gather psychometric topics
            if GATHER_PSYCHOMETRIC_TOPICS:
                score_columns_dict = {col: [row[col]] for col in score_columns if not col.startswith('psychometric_heb_') and pd.notna(row[col])}
                if 'psychometric_heb_math_score' in row and pd.notna(row['psychometric_heb_math_score']):
                    score_columns_dict['psychometric_heb_math_score'] = [row['psychometric_heb_math_score']]
            
                if all(col in row and pd.notna(row[col]) for col in heb_benchmarks):
                    score_columns_dict['psychometric_heb_hebrew_score'] = [row[col] for col in heb_benchmarks]
                if all(col in row and pd.notna(row[col]) for col in eng_benchmarks):
                    score_columns_dict['psychometric_heb_english_score'] = [row[col] for col in eng_benchmarks]
            else:
                score_columns_dict = {}
                for j, score_col in enumerate(score_columns):
                    score = row[score_col]
                    if pd.notna(score):
                        score_columns_dict[benchmark_names[j]] = [score]
                
            if score_columns_dict:  # Only add if there are valid scores
                scores = [np.mean(x) for x in list(score_columns_dict.values())]
                trace_config = {
                    'x': list(score_columns_dict.keys()),
                    'y': scores,
                    'mode': mode,
                    'name': f'{model_name} ({run_time})',
                    'marker': dict(size=10 if mode == 'markers' else 8, color=colors[i % len(colors)]),
                    'text': [f'{score:.3f}' for score in scores],
                    'textposition': 'top center',
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                    'Benchmark: %{x}<br>' +
                                    'Score: %{y:.3f}<extra></extra>'
                }
                
                # Add line configuration only if in line mode
                if mode == 'lines+markers':
                    trace_config['line'] = dict(color=colors[i % len(colors)], width=2)
                
                fig.add_trace(go.Scatter(**trace_config))
        
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

def handle_cell_click(row_idx: int, col_name: str, df: pd.DataFrame):
    """
    Handle cell click event and show detailed viewer in the same interface
    """
    # Check if clicked column is a score column
    if not col_name.endswith('_score'):
        return None, gr.update(visible=False)
    
    try:
        # Get row data
        row_data = df.iloc[row_idx].to_dict()
        
        # Get parquet path
        parquet_path = f'{scores_sum_directory}{df.at[row_idx, col_name.replace("_score","_details")]}'
        print(f"Opening parquet file: {parquet_path}")
        
        local_temp_dir = os.path.join(local_save_directory, 'temp_parquets')
        os.makedirs(local_temp_dir, exist_ok=True)
        local_file_path = os.path.join(local_temp_dir, os.path.basename(parquet_path))
        print(f"Downloading from S3: {parquet_path} to {local_file_path}")
        try:
            download_s3_file(parquet_path, local_file_path)
        except Exception as e:
            print(f"Failed to download parquet file from S3: {e}")
            return gr.update(value=pd.DataFrame({'Error': [f"Failed to download data file: {e}"]})), gr.update(visible=True)
        
        parquet_path = local_file_path
        print(f"Using local parquet file: {parquet_path}")
        
        # Load the data directly

        detailed_df = create_detailed_viewer(parquet_path, "Simple")
        
        return detailed_df, gr.update(visible=True)
        
    except Exception as e:
        print(f"Error handling cell click: {e}")
        traceback.print_exc()
        return gr.update(value=pd.DataFrame({'Error': [str(e)]})), gr.update(visible=True)
    
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
                value=lambda: df_state.value[[col for col in df_state.value.columns if not col.endswith('_details')]],
                wrap=True,
                max_height=600,
                show_fullscreen_button=True,
                interactive=False
            )
            
            def update_col_choices():
                df, score_cols = make_clickable_table()
                return gr.Dropdown(choices=score_cols)
            
            def refresh_table():
                generate_runs_table()  # Refresh data
                df = create_data_table()
                _, score_cols = make_clickable_table()
                df = df[[col for col in df_state.value.columns if not col.endswith('_details')]]
                return df
            
            refresh_data_btn = gr.Button("🔄 Refresh Data", variant="secondary")
            
        #  Tab for detailed table view with selection

            refresh_data_btn.click(
                fn=refresh_table,
                outputs=[data_table]
            )
            
       
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
            
            # Add plot mode toggle
            with gr.Row():
                plot_mode = gr.Radio(
                    choices=["Lines + Markers", "Markers Only"],
                    value="Lines + Markers",
                    label="Plot Style",
                    scale=2
                )
                gr.Markdown("")  # Spacer
            
            comparison_plot = gr.Plot(value=create_benchmark_comparison(["Latest per model"], "Lines + Markers"))
            
            # Update chart when selection changes
            run_selector.change(
                fn=create_benchmark_comparison,
                inputs=[run_selector, plot_mode],
                outputs=comparison_plot
            )
            
            # Update chart when plot mode changes
            plot_mode.change(
                fn=create_benchmark_comparison,
                inputs=[run_selector, plot_mode],
                outputs=comparison_plot
            )
            
            with gr.Row():
                refresh_comparison_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
                refresh_runs_btn = gr.Button("🔄 Refresh Run List", variant="secondary")
            
            refresh_comparison_btn.click(
                fn=create_benchmark_comparison,
                inputs=[run_selector, plot_mode],
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
        
        with gr.Tab("🔍 Detailed View"):
            gr.Markdown("### View Detailed Results for Specific Dataset")
            
            with gr.Row():
                run_selector = gr.Dropdown(
                    label="run_id",
                    choices=[(name, idx) for idx, name in enumerate(df_state.value.model_name.values.tolist())], 
                    value=None
                )
                col_selector = gr.Dropdown(
                    label="Dataset Column",
                    choices=[],
                    value=None
                )
                view_btn = gr.Button("🔍 View Details", variant="primary")
            
            # View mode selector
            view_mode_selector = gr.Radio(
                choices=["Simple", "Full"],
                value="Simple",
                label="View Mode",
                info="Simple: Shows query, accuracy, and model output | Full: Shows all columns",
                visible=False
            )
            
            # Results display
            detailed_results = gr.Dataframe(
                wrap=True,
                max_height=600,
                interactive=False,
                line_breaks=True,
                visible=False
            )
            
            # Stats display
            stats_display = gr.Markdown(visible=False)
            
            # Download button
            with gr.Row():
                download_btn = gr.Button("💾 Download Current View as CSV", variant="secondary", visible=False)
                refresh_detail_btn = gr.Button("🔄 Refresh View", variant="secondary", visible=False)
            
            download_output = gr.File(label="Download CSV", visible=False)
            
            def on_view_click(row_idx, col_name, df):
                if col_name is None:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,
                        "Please select a dataset column"
                    )
                try:
                    if isinstance(row_idx, str):
                        row_name = row_idx
                        if row_name in df.model_name.tolist():
                            row_idx = df.model_name.tolist().index(row_name)
                        else:
                            return (
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                None,
                                f"Model name '{row_name}' not found"
                            )
                    
                    # Get parquet path
                    parquet_path = f'{scores_sum_directory}{df.at[int(row_idx), col_name.replace("_score","_details")]}'
                    print(f"Opening parquet file: {parquet_path}")
                    
                    local_temp_dir = os.path.join(local_save_directory, 'temp_parquets')
                    os.makedirs(local_temp_dir, exist_ok=True)
                    local_file_path = os.path.join(local_temp_dir, os.path.basename(parquet_path))
                    
                    try:
                        download_s3_file(parquet_path, local_file_path)
                    except Exception as e:
                        print(f"Failed to download parquet file from S3: {e}")
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            None,
                            f"Failed to download data file: {e}"
                        )
                    
                    parquet_path = local_file_path
                    
                    # Load the data
                    from detailed_results_viewer import create_detailed_viewer
                    detailed_df = create_detailed_viewer(parquet_path, "Simple")
                    
                    # Calculate stats
                    if 'Error' in detailed_df.columns or 'Message' in detailed_df.columns:
                        stats = "No statistics available"
                    else:
                        stats = f"**Total Samples:** {len(detailed_df)} "
                        stats += f"**Columns Displayed:** {len(detailed_df.columns)} "
                        
                        if 'Correct' in detailed_df.columns:
                            correct_count = (detailed_df['Correct'] == '✓').sum()
                            total_count = len(detailed_df)
                            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
                            stats += f"**Correct:** {correct_count}/{total_count} ({accuracy:.2f}%) "
                        
                        if 'Accuracy' in detailed_df.columns:
                            avg_acc = detailed_df['Accuracy'].mean()
                            if not pd.isna(avg_acc):
                                stats += f"**Average Accuracy:** {avg_acc:.4f} "
                    
                    # Store parquet path in a state for view mode changes
                    return (
                        gr.update(value=detailed_df, visible=True),
                        gr.update(value=stats, visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),  # download button
                        gr.update(visible=True),  # refresh button
                        gr.update(visible=False), # download output (hide until clicked)
                        parquet_path,
                        f"Loaded {len(detailed_df)} samples from {col_name}"
                    )
                    
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,
                        f"Error: {str(e)}"
                    )
            
            # State to store current parquet path
            current_parquet_path = gr.State(value=None)
            info_box = gr.Textbox(label="Status", interactive=False)
            
            view_btn.click(
                fn=on_view_click,
                inputs=[run_selector, col_selector, df_state],
                outputs=[detailed_results, stats_display, view_mode_selector, 
                        download_btn, refresh_detail_btn, download_output, 
                        current_parquet_path, info_box]
            )
            
            # Handle view mode changes
            def change_view_mode(view_mode, parquet_path):
                if parquet_path is None:
                    return gr.update(), gr.update()
                
                from detailed_results_viewer import create_detailed_viewer
                detailed_df = create_detailed_viewer(parquet_path, view_mode)
                
                # Recalculate stats
                if 'Error' in detailed_df.columns or 'Message' in detailed_df.columns:
                    stats = "No statistics available"
                else:
                    stats = f"**Total Samples:** {len(detailed_df)} "
                    stats += f"**Columns Displayed:** {len(detailed_df.columns)} "
                    
                    if 'Correct' in detailed_df.columns:
                        correct_count = (detailed_df['Correct'] == '✓').sum()
                        total_count = len(detailed_df)
                        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
                        stats += f"**Correct:** {correct_count}/{total_count} ({accuracy:.2f}%) "
                    
                    if 'Accuracy' in detailed_df.columns:
                        avg_acc = detailed_df['Accuracy'].mean()
                        if not pd.isna(avg_acc):
                            stats += f"**Average Accuracy:** {avg_acc:.4f} "
                
                return gr.update(value=detailed_df), gr.update(value=stats)
            
            view_mode_selector.change(
                fn=change_view_mode,
                inputs=[view_mode_selector, current_parquet_path],
                outputs=[detailed_results, stats_display]
            )
            
            # Handle download button
            def download_csv(view_mode, parquet_path):
                """Generate CSV file and return the file path"""
                if parquet_path is None:
                    error_path = "/tmp/error.txt"
                    with open(error_path, 'w') as f:
                        f.write("No data loaded. Please view details first.")
                    return gr.update(value=error_path, visible=True)
                
                try:
                    from detailed_results_viewer import create_detailed_viewer
                    df = create_detailed_viewer(parquet_path, view_mode)
                    
                    # Generate unique filename with timestamp
                    import time
                    timestamp = int(time.time())
                    dataset_name = os.path.basename(parquet_path).replace('.parquet', '').replace('details_', '')
                    filename = f"detailed_view_{dataset_name}_{timestamp}.csv"
                    temp_path = os.path.join("/tmp", filename)
                    
                    # Save to CSV
                    df.to_csv(temp_path, index=False, encoding='utf-8')
                    
                    print(f"CSV saved to: {temp_path}")
                    return gr.update(value=temp_path, visible=True)
                except Exception as e:
                    print(f"Error creating CSV: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return error file
                    error_path = "/tmp/error.txt"
                    with open(error_path, 'w') as f:
                        f.write(f"Error creating CSV: {str(e)}")
                    return gr.update(value=error_path, visible=True)
            
            download_btn.click(
                fn=download_csv,
                inputs=[view_mode_selector, current_parquet_path],
                outputs=download_output
            )
            
            # Handle refresh button
            def refresh_view(view_mode, parquet_path):
                if parquet_path is None:
                    return gr.update(), gr.update(), "No data loaded"
                
                try:
                    from detailed_results_viewer import create_detailed_viewer
                    detailed_df = create_detailed_viewer(parquet_path, view_mode)
                    
                    # Recalculate stats
                    if 'Error' in detailed_df.columns or 'Message' in detailed_df.columns:
                        stats = "No statistics available"
                    else:
                        stats = f"**Total Samples:** {len(detailed_df)} "
                        stats += f"**Columns Displayed:** {len(detailed_df.columns)} "
                        
                        if 'Correct' in detailed_df.columns:
                            correct_count = (detailed_df['Correct'] == '✓').sum()
                            total_count = len(detailed_df)
                            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
                            stats += f"**Correct:** {correct_count}/{total_count} ({accuracy:.2f}%) "
                        
                        if 'Accuracy' in detailed_df.columns:
                            avg_acc = detailed_df['Accuracy'].mean()
                            if not pd.isna(avg_acc):
                                stats += f"**Average Accuracy:** {avg_acc:.4f} "
                    
                    return gr.update(value=detailed_df), gr.update(value=stats), "View refreshed"
                except Exception as e:
                    return gr.update(), gr.update(), f"Error refreshing: {str(e)}"
            
            refresh_detail_btn.click(
                fn=refresh_view,
                inputs=[view_mode_selector, current_parquet_path],
                outputs=[detailed_results, stats_display, info_box]
            )
            
            demo.load(fn=update_col_choices, outputs=col_selector)  
PORT = 7680
# Launch the app
if __name__ == "__main__":
    print("Starting Gradio app...")
    # generate_runs_table()  # Initial data generation
    check_port_or_raise(PORT, timeout=3,auto_kill=True, retry=True)
    demo.launch(
        share=False,
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
