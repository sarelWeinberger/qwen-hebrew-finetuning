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
from extract_benchmark_results import summarize_benchmark_runs, rename_benchmark_columns, clean_model_name
from check_port_avaliablity import check_port_or_raise
import webbrowser
import threading
from typing import List, Optional, Dict
from dashboard.detailed_results_viewer import create_detailed_viewer   
from dashboard.dashboard_graphs import create_training_progress_plot, create_score_over_time, create_psychometric_detailed, create_model_performance_heatmap, create_benchmark_comparison
from dashboard.config import Config
from dashboard.utils import load_data
from dashboard.detailed_results_viewer import create_detailed_viewer
from dashboard.config import Config

print("Starting Gradio app code...")

def generate_runs_table():
    # Use the original local path as default
    
    print("Summarizing benchmark runs from:", Config.scores_sum_directory)
    # You can specify a custom local directory to save the CSV
    summarize_benchmark_runs(Config.scores_sum_directory, Config.local_save_directory,Config.csv_filename)
    

def get_available_run_directories(df):
    """Get list of unique base model directories (run directories)"""
    if df is None or df.empty:
        return []
    # use model_group when available and model_name otherwise
    get_model_group_or_name = lambda x: x['model_group'] if 'model_group' in x and pd.notna(x['model_group']) and x['model_group'] != "" else x['model_name']
    run_dirs = df.apply(get_model_group_or_name, axis=1).unique().tolist()
    return run_dirs

def get_checkpoints_for_run(df, run_directory):
    """Get all checkpoints for a specific run directory"""
    if df is None or df.empty or not run_directory:
        return []
    
    # Find all models that start with this run directory
    matching_models = df[df['model_name'].str.startswith(run_directory)]
    return matching_models['model_name'].tolist()

def filter_partial_training(df, include_partial):
    """Filter dataframe based on partial training toggle"""
    if include_partial:
        return df
    
    # Filter out rows where samples_number < 1000
    if 'samples_number' in df.columns:
        # Keep rows where samples_number is None or > 1000
        df = df[(df['samples_number'].isna()) | (df['samples_number'] > 1000)]
    
    return df

def get_available_runs(df):
    """Get list of available runs for the dropdown"""
    try:
        # df, _ = load_data()
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

def create_data_table(include_partial = False):
    try:
        
        df, _ = load_data()
        # Format timestamp for display
        df_display = df.copy()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display = filter_partial_training(df_display, include_partial)
        return df_display
    except Exception as e:
        print(f"Error creating data table: {e}")
        #print traceback
        traceback.print_exc()

        # Return empty dataframe with error message
        return pd.DataFrame({'Error': [f'Failed to load data: {str(e)}']})


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
        parquet_path = f'{Config.scores_sum_directory}{df.at[row_idx, col_name.replace("_score","_details")]}'
        print(f"Opening parquet file: {parquet_path}")
        
        local_temp_dir = os.path.join(Config.local_save_directory, 'temp_parquets')
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

with gr.Blocks(title="Benchmark Results Visualization", theme=gr.themes.Soft(), css="""
    .small-toggle label {
        font-size: 0.75rem !important;
        opacity: 0.6;
    }
    .small-toggle input {
        transform: scale(0.8);
    }
""") as demo:
    gr.Markdown("# 🏆 Benchmark Results Visualization Dashboard")
    gr.Markdown("Interactive visualization of Hebrew language model benchmark results")
    
    with gr.Tabs():
        with gr.Tab("📊 Data Table"):
            gr.Markdown("### Raw Benchmark Data")
            # gr.Markdown("**💡 Tip:** Click on any score cell to view detailed results for that dataset")
            
            # Create state to store dataframe
            df_state = gr.State(value=create_data_table())
            # State for partial training toggle
            partial_training_state = gr.State(value=False)

            data_table = gr.Dataframe(
                value=lambda: df_state.value[[col for col in df_state.value.columns if not col.endswith('_details')]].round(3),
                wrap=True,
                max_height=600,
                show_fullscreen_button=True,
                interactive=False
            )
            
            def update_col_choices():
                df, score_cols = make_clickable_table()
                return gr.Dropdown(choices=score_cols)
            
            def refresh_table(include_partial):
                generate_runs_table()  # Refresh data
                df = create_data_table(include_partial)
                _, score_cols = make_clickable_table()
                df = df[[col for col in df.columns if not col.endswith('_details')]]
                return df
            
            def toggle_partial_training(include_partial, df):
                """Filter table based on partial training toggle"""
                df_full = create_data_table()
                df_filtered = filter_partial_training(df_full, include_partial)
                df_filtered = df_filtered[[col for col in df_filtered.columns if not col.endswith('_details')]]
                return df_filtered
            
            with gr.Row():
                refresh_data_btn = gr.Button("🔄 Refresh Data", variant="secondary")
                gr.Markdown("")  # Spacer
            
            # Small toggle at the bottom for partial training
            with gr.Row():
                gr.Markdown("")  # Spacer to push toggle to the right
                partial_training_toggle = gr.Checkbox(
                    label="Include partial runs (DEBUG)",
                    value=False,
                    scale=0,
                    container=False,
                    elem_classes=["small-toggle"]
                )
        #  Tab for detailed table view with selection

            refresh_data_btn.click(
                fn=refresh_table,
                inputs=[partial_training_toggle],
                outputs=[data_table]
            )
            
            partial_training_toggle.change(
                fn=toggle_partial_training,
                inputs=[partial_training_toggle, df_state],
                outputs=[data_table]
            )
            
       
        with gr.Tab("📈 Benchmark Comparison"):
            gr.Markdown("### Compare Scores Across Benchmarks")
            gr.Markdown("Select specific runs to compare, or use 'Latest per model' to compare the most recent results for each model.")
            
            # Run selection dropdown
            run_selector = gr.Dropdown(
                choices=get_available_runs(df_state.value),
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
            
            comparison_plot = gr.Plot(value=create_benchmark_comparison(["Latest per model"],df_state.value, "Lines + Markers"))
            
            # Update chart when selection changes
            run_selector.change(
                fn=create_benchmark_comparison,
                inputs=[run_selector,df_state , plot_mode],
                outputs=comparison_plot
            )
            
            # Update chart when plot mode changes
            plot_mode.change(
                fn=create_benchmark_comparison,
                inputs=[run_selector,df_state , plot_mode],
                outputs=comparison_plot
            )
            
            with gr.Row():
                refresh_comparison_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
                refresh_runs_btn = gr.Button("🔄 Refresh Run List", variant="secondary")
            
            refresh_comparison_btn.click(
                fn=create_benchmark_comparison,
                inputs=[run_selector, df_state, plot_mode],
                outputs=comparison_plot
            )
            refresh_runs_btn.click(
                fn=get_available_runs,
                inputs=[df_state],
                outputs=run_selector
            )
        with gr.Tab("📉 Training Progress"):
            gr.Markdown("### Benchmark Scores Over Training Steps")
            gr.Markdown("Compare how benchmark scores evolve during training across different runs and checkpoints.")
            

            available_run_dirs = get_available_run_directories(df_state.value)
            available_benchmarks = sorted([col for col in df_state.value.columns if col.endswith('_score')])
            
            with gr.Row():
                run_dir_selector = gr.Dropdown(
                    choices=available_run_dirs,
                    value=[available_run_dirs[0]] if available_run_dirs else [],
                    multiselect=True,
                    label="Select Run Directories",
                    info="Choose which training runs to compare"
                )
                
                benchmark_selector = gr.Dropdown(
                    choices=['All', 'Avg Score', *available_benchmarks],
                    value='Avg Score',
                    multiselect=True,
                    label="Select Benchmarks",
                    info="Choose which benchmarks to display"
                )
            
                with gr.Row():
                    include_checkpoints_toggle = gr.Checkbox(
                        label="Include all checkpoints",
                        value=True,
                        info="When enabled, shows all checkpoints. When disabled, shows only base model."
                    )
                    subplot_per_benchmark = gr.Checkbox(
                        label="Use subplots for each benchmark",
                        value=False,
                        info="Display each benchmark in its own subplot",
                    )
            
            progress_plot = gr.Plot(
                value=create_training_progress_plot(
                    df_state.value, 
                    [available_run_dirs[0]] if available_run_dirs else [],
                    ['Avg Score'],
                    False
                )
            )
            
            # Update plot when selections change
            def update_progress_plot(run_dirs, benchmarks, include_checkpoints, subplot_per_benchmark):
                return create_training_progress_plot(df_state.value, run_dirs, benchmarks, include_checkpoints , subplot_per_benchmark)
            
            run_dir_selector.change(
                fn=update_progress_plot,
                inputs=[run_dir_selector, benchmark_selector, include_checkpoints_toggle, subplot_per_benchmark],

                outputs=progress_plot
            )
            
            benchmark_selector.change(
                fn=update_progress_plot,
                inputs=[run_dir_selector, benchmark_selector, include_checkpoints_toggle, subplot_per_benchmark],
                outputs=progress_plot
            )
            
            include_checkpoints_toggle.change(
                fn=update_progress_plot,
                inputs=[run_dir_selector, benchmark_selector, include_checkpoints_toggle, subplot_per_benchmark],
                outputs=progress_plot
            )
            subplot_per_benchmark.change(
                fn=update_progress_plot,
                inputs=[run_dir_selector, benchmark_selector, include_checkpoints_toggle, subplot_per_benchmark],
                outputs=progress_plot
            )
            
            with gr.Row():
                refresh_progress_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
                refresh_progress_lists_btn = gr.Button("🔄 Refresh Lists", variant="secondary")
            
            def refresh_progress_data():
                df, score_columns = load_data()
                run_dirs = get_available_run_directories(df)
                benchmarks = sorted([col for col in score_columns if col.endswith('_score')])
                return (
                    gr.update(choices=run_dirs),
                    gr.update(choices=benchmarks)
                )
            
            refresh_progress_btn.click(
                fn=update_progress_plot,
                # inputs=[run_dir_selector, benchmark_selector, include_checkpoints_toggle],
                inputs=[run_dir_selector, benchmark_selector],
                outputs=progress_plot
            )
            
            refresh_progress_lists_btn.click(
                fn=refresh_progress_data,
                outputs=[run_dir_selector, benchmark_selector]
            )
           
        with gr.Tab("⏰ Scores Over Time"):
            gr.Markdown("### Performance Trends Over Time")
            time_plot = gr.Plot(value=create_score_over_time())
            
            refresh_time_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_time_btn.click(fn=create_score_over_time, outputs=time_plot)
        
        with gr.Tab("🎯 Psychometric Details"):
            gr.Markdown("### Detailed Psychometric Benchmark Analysis")
            psychometric_plot = gr.Plot(value=create_psychometric_detailed(df_state.value))
            
            refresh_psychometric_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_psychometric_btn.click(fn=create_psychometric_detailed,inputs=[df_state], outputs=psychometric_plot)
        
        with gr.Tab("🔥 Performance Heatmap"):
            gr.Markdown("### Model Performance Heatmap")
            heatmap_plot = gr.Plot(value=create_model_performance_heatmap())
            
            refresh_heatmap_btn = gr.Button("🔄 Refresh Chart", variant="secondary")
            refresh_heatmap_btn.click(fn=create_model_performance_heatmap, outputs=heatmap_plot)
        
        with gr.Tab("🔍 Detailed View"):
            gr.Markdown("### View Detailed Results for Specific Dataset")
            
            with gr.Row():
                run_selector_det = gr.Dropdown(
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
                    parquet_path = f'{Config.scores_sum_directory}{df.at[int(row_idx), col_name.replace("_score","_details")]}'
                    print(f"Opening parquet file: {parquet_path}")
                    
                    local_temp_dir = os.path.join(Config.local_save_directory, 'temp_parquets')
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
                inputs=[run_selector_det, col_selector, df_state],
                outputs=[detailed_results, stats_display, view_mode_selector, 
                        download_btn, refresh_detail_btn, download_output, 
                        current_parquet_path, info_box]
            )
            
            # Handle view mode changes
            def change_view_mode(view_mode, parquet_path):
                if parquet_path is None:
                    return gr.update(), gr.update()

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
## gradio logs 
# sudo tail -n 100 /var/log/gradio.log
# kill previous process on port 7680
# lsof -ti:7680 | xargs kill -9 2>/dev/null || echo "No process found on port 7680"

# check if the protocol is avaliable
# netstat -tlnp | grep 7680
# start server in background
# sudo systemctl stop gradio
# sudo systemctl daemon-reload
# sudo systemctl enable gradio
# sudo systemctl start gradio
## check status
# sudo systemctl status gradio
## monitor logs
# sudo journalctl -u gradio -f
## stop server
# sudo systemctl stop gradio
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
#         psychometric_cols = [col for col in score_columns if col.startswith('Ψ')]
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
#         psychometric_cols = [col for col in score_columns if col.startswith('Ψ')]
        
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
#                     categories.append(col.replace('Ψ_', '').replace('_score', '').replace('_', ' ').title())
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
