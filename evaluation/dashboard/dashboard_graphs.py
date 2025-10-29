import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from extract_benchmark_results import summarize_benchmark_runs, rename_benchmark_columns, clean_model_name
from dashboard.utils import load_data, get_base_model_name


def create_training_progress_plot(df,
                                  selected_runs,
                                  selected_benchmarks,
                                  include_checkpoints=True,
                                  subplot_per_benchmark=False):
    """Create a plot showing benchmark scores over training steps"""
    try:
        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        if not selected_runs:
            fig = go.Figure()
            fig.add_annotation(text="Please select run directories", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Handle special benchmark selections
        all_benchmarks = [col for col in df.columns if col.endswith('_score') and not col.endswith('_std')]
        
        if not selected_benchmarks or 'All' in selected_benchmarks:
            benchmarks_to_plot = all_benchmarks
        elif 'Avg Score' in selected_benchmarks:
            # Calculate average score across all benchmarks
            if 'avg_score' not in df.columns:
                df['avg_score'] = df[all_benchmarks].mean(axis=1, skipna=True)
            benchmarks_to_plot = ['avg_score']
        else:
            benchmarks_to_plot = selected_benchmarks
        
        # Filter out avg_score from benchmarks if it's already there
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in df.columns]
        
        if not benchmarks_to_plot:
            fig = go.Figure()
            fig.add_annotation(text="No valid benchmarks selected", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Debug: Print what we're looking for
        print(f"Selected runs: {selected_runs}")
        print(f"Benchmarks to plot: {benchmarks_to_plot}")
        
        # Calculate the global step range for horizontal lines
        global_min_step = float('inf')
        global_max_step = float('-inf')
        
        for run_dir in selected_runs:
            # Get all models for this run
            run_models = df[df['model_name'].apply(get_base_model_name) == run_dir].copy()
            
            # Debug print
            print(f"\nRun: {run_dir}")
            print(f"Found {len(run_models)} models")
            if len(run_models) > 0:
                print("Model names:", run_models['model_name'].tolist())
                print("Steps:", run_models['steps'].tolist())
            
            if not run_models.empty:
                # Handle steps
                run_models['step'] = run_models['steps']
                
                # For models without steps, check if it's a base model
                base_mask = run_models['step'].isna()
                if base_mask.any():
                    # Check if model_name doesn't contain step information
                    for idx in run_models[base_mask].index:
                        if '/' not in str(run_models.loc[idx, 'model_name']) or \
                           not any(step_word in str(run_models.loc[idx, 'model_name']) 
                                  for step_word in ['step-', 'checkpoint-']):
                            run_models.loc[idx, 'step'] = 0
                
                steps = run_models['step'].dropna()
                if len(steps) > 0:
                    global_min_step = min(global_min_step, steps.min())
                    global_max_step = max(global_max_step, steps.max())
        
        # If no valid steps found, use default range
        if global_min_step == float('inf'):
            global_min_step = 0
            global_max_step = 1000
        
        print(f"\nGlobal step range: {global_min_step} to {global_max_step}")
        
        # Create figure with subplots if requested
        if subplot_per_benchmark and len(benchmarks_to_plot) > 1:
            n_subplots = len(benchmarks_to_plot)
            n_cols = min(2, n_subplots)  # Max 2 columns
            n_rows = (n_subplots + n_cols - 1) // n_cols
            
            # Calculate appropriate vertical spacing
            if n_rows > 1:
                max_spacing = 1 / (n_rows - 1)
                vertical_spacing = min(0.1, max_spacing * 0.8)  # Use 80% of max allowed
            else:
                vertical_spacing = 0.1
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[b.replace('_score', '').replace('_', ' ').title() 
                               if b != 'avg_score' else 'Average Score' 
                               for b in benchmarks_to_plot],
                vertical_spacing=vertical_spacing,
                horizontal_spacing=0.1
            )
        else:
            fig = go.Figure()
        
        # Expanded color palette and symbols
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                   'pentagon', 'hexagon', 'star', 'hexagram', 'octagon']
        
        for run_idx, run_dir in enumerate(selected_runs):
            # Get all models for this run
            run_models = df[df['model_name'].apply(get_base_model_name) == run_dir].copy()
            
            if run_models.empty:
                continue
            
            # Extract step numbers
            run_models['step'] = run_models['steps']
            
            # Separate base model and checkpoints
            base_model = run_models[run_models['step'].isna()].copy()
            checkpoint_models = run_models[run_models['step'].notna()].copy()
            
            # For base models, set step to 0 if they don't have checkpoint in name
            if not base_model.empty:
                for idx in base_model.index:
                    if '/' not in str(base_model.loc[idx, 'model_name']) or \
                       not any(step_word in str(base_model.loc[idx, 'model_name']) 
                              for step_word in ['step-', 'checkpoint-']):
                        base_model.loc[idx, 'step'] = 0
            
            if include_checkpoints:
                # Include all checkpoints
                if not checkpoint_models.empty:
                    # Use checkpoints
                    run_models_to_plot = checkpoint_models
                    # Also add base model if it exists and has step 0
                    if not base_model.empty and 0 in base_model['step'].values:
                        run_models_to_plot = pd.concat([base_model[base_model['step'] == 0], checkpoint_models])
                elif not base_model.empty:
                    # No checkpoints, use base model
                    run_models_to_plot = base_model
                else:
                    continue
            else:
                # Only include base model
                if base_model.empty:
                    continue
                run_models_to_plot = base_model[base_model['step'] == 0]
            
            if run_models_to_plot.empty:
                continue
            
            # Remove any remaining NaN steps
            run_models_to_plot = run_models_to_plot[run_models_to_plot['step'].notna()]
            
            # Sort by step
            run_models_to_plot = run_models_to_plot.sort_values('step')
            
            # Take the latest run for each step (in case of duplicates)
            run_models_to_plot = run_models_to_plot.groupby('step').last().reset_index()
            
            print(f"\nPlotting {len(run_models_to_plot)} points for {run_dir}")
            print(f"Steps: {run_models_to_plot['step'].tolist()}")
            
            # Determine if this is a single checkpoint run
            is_single_checkpoint = len(run_models_to_plot) == 1
            
            # Assign color for this run (consistent across benchmarks)
            run_color = colors[run_idx % len(colors)]
            
            # Plot each selected benchmark
            for bench_idx, benchmark in enumerate(benchmarks_to_plot):
                if benchmark not in run_models_to_plot.columns:
                    continue
                
                # Get scores for this benchmark
                scores = run_models_to_plot[benchmark].values
                steps = run_models_to_plot['step'].values
                
                # Filter out NaN values
                valid_mask = ~pd.isna(scores)
                scores = scores[valid_mask]
                steps = steps[valid_mask]
                
                if len(scores) == 0:
                    continue
                
                # Create trace name
                if benchmark == 'avg_score':
                    benchmark_display = 'Average Score'
                else:
                    benchmark_display = benchmark.replace('_score', '').replace('_', ' ').title()
                
                # Choose symbol and line style
                symbol = symbols[bench_idx % len(symbols)]
                line_style = line_styles[bench_idx % len(line_styles)]
                
                # For single checkpoint runs, create a horizontal line
                if is_single_checkpoint:
                    # Create horizontal line across the full step range
                    original_step = steps[0]
                    original_score = scores[0]
                    
                    # Create horizontal line
                    steps = [global_min_step, original_step, global_max_step]
                    scores = [original_score, original_score, original_score]
                    
                    # Use solid line for single checkpoint runs
                    line_style = 'solid'
                
                # Determine subplot position
                if subplot_per_benchmark and len(benchmarks_to_plot)> 1:
                    row = (bench_idx // 2) + 1
                    col = (bench_idx % 2) + 1
                    
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines+markers' if not is_single_checkpoint else 'lines',
                        name=f"{run_dir}",
                        line=dict(color=run_color, width=2, dash=line_style),
                        marker=dict(size=8, color=run_color, symbol=symbol) if not is_single_checkpoint else dict(size=10, color=run_color, symbol=symbol),
                        legendgroup=run_dir,
                        showlegend=(bench_idx == 0),  # Only show legend for first benchmark
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    f'Benchmark: {benchmark_display}<br>' +
                                    'Step: %{x}<br>' +
                                    'Score: %{y:.3f}<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;extra&gt;</span><span style="color: black; font-weight: normal;"></span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/extra&gt;</span><br><br></span>'
                    ), row=row, col=col)
                    
                    # Add a marker at the original step for single checkpoint
                    if is_single_checkpoint:
                        fig.add_trace(go.Scatter(
                            x=[original_step],
                            y=[original_score],
                            mode='markers',
                            marker=dict(size=12, color=run_color, symbol=symbol),
                            showlegend=False,
                            legendgroup=run_dir,
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        f'Benchmark: {benchmark_display}<br>' +
                                        'Step: %{x}<br>' +
                                        'Score: %{y:.3f}<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;extra&gt;</span><span style="color: black; font-weight: normal;"></span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/extra&gt;</span><br><br></span>'
                        ), row=row, col=col)
                else:
                    trace_name = f"{run_dir} - {benchmark_display}"
                    
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines+markers' if not is_single_checkpoint else 'lines',
                        name=trace_name,
                        line=dict(color=run_color, width=2, dash=line_style),
                        marker=dict(size=8, color=run_color, symbol=symbol) if not is_single_checkpoint else dict(size=10, color=run_color, symbol=symbol),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Step: %{x}<br>' +
                                    'Score: %{y:.3f}<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;extra&gt;</span><span style="color: black; font-weight: normal;"></span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/extra&gt;</span><br><br></span>'
                    ))
                    
                    # Add a marker at the original step for single checkpoint
                    if is_single_checkpoint:
                        fig.add_trace(go.Scatter(
                            x=[original_step],
                            y=[original_score],
                            mode='markers',
                            name=f"{trace_name} (checkpoint)",
                            marker=dict(size=12, color=run_color, symbol=symbol),
                            showlegend=False,
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Step: %{x}<br>' +
                                        'Score: %{y:.3f}<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;extra&gt;</span><span style="color: black; font-weight: normal;"></span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/extra&gt;</span><br><br></span>'
                        ))
        
        # Update layout
        title = 'Training Progress: Benchmark Scores Over Steps'
        if 'Avg Score' in selected_benchmarks:
            title = 'Training Progress: Average Score Over Steps'
        
        if subplot_per_benchmark and len(benchmarks_to_plot)> 1:
            fig.update_xaxes(title_text='Training Steps', showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(title_text='Score', showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Calculate appropriate height based on number of rows
            height_per_row = 350
            total_height = min(height_per_row * n_rows, 1200)  # Cap at 1200px
            
            fig.update_layout(
                title=title,
                height=total_height,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                margin=dict(r=150)
            )
        else:
            fig.update_layout(
                title=title,
                xaxis_title='Training Steps',
                yaxis_title='Score',
                height=600,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                margin=dict(r=250),
                hovermode='closest'
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        print(f"Error creating training progress plot: {e}")
        import traceback
        traceback.print_exc()
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    
def create_benchmark_comparison(selected_runs, df=None, plot_mode="Lines + Markers"):
    try:
        if df is None:
            df, score_columns = load_data()
        else:
            score_columns = [col for col in df.columns if col.endswith('_score')]
        # sort score columns alphabetically
        score_columns = sorted(score_columns)
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
            run_time = row['timestamp'].strftime('%Y-%m-%d %H:%M') if not isinstance(row['timestamp'], str) else row['timestamp']
            heb_benchmarks = ["Ψ_understanding_hebrew_score",
                              "Ψ_sentence_text_hebrew_score",
                              "Ψ_sentence_complete_hebrew_score",
                              "Ψ_analogies_hebrew_score"]
            eng_benchmarks = ["Ψ_restatement_english_score",
                              "Ψ_sentence_text_english_score",
                              "Ψ_sentence_complete_english_score"]
            GATHER_PSYCHOMETRIC_TOPICS = True
            #change score_columns to gather psychometric topics
            if GATHER_PSYCHOMETRIC_TOPICS:
                score_columns_dict = {col: [row[col]] for col in score_columns if not col.startswith('Ψ_') and pd.notna(row[col])}
                if 'Ψ_math_score' in row and pd.notna(row['Ψ_math_score']):
                    score_columns_dict['Ψ_math_score'] = [row['Ψ_math_score']]
            
                if all(col in row and pd.notna(row[col]) for col in heb_benchmarks):
                    score_columns_dict['Ψ_hebrew_score'] = [row[col] for col in heb_benchmarks]
                if all(col in row and pd.notna(row[col]) for col in eng_benchmarks):
                    score_columns_dict['Ψ_english_score'] = [row[col] for col in eng_benchmarks]
            else:
                score_columns_dict = {}
                for j, score_col in enumerate(score_columns):
                    score = row[score_col]
                    if pd.notna(score):
                        score_columns_dict[benchmark_names[j]] = [score]
            # make a lists of scores and score in the order of benchmark_names
            score_columns_dict = {k.replace('_score', '').replace('_', ' '): v for k, v in score_columns_dict.items()}

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
        psychometric_cols = [col for col in score_columns if col.startswith('Ψ')]
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


def create_psychometric_detailed(df = None):
    try:
        if df is None:
            df, score_columns = load_data()
        else:
            score_columns = [col for col in df.columns if col.endswith('_score')]
        
        # Get psychometric columns
        psychometric_cols = [col for col in score_columns if col.startswith('Ψ')]
        
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
                    categories.append(col.replace('Ψ_', '').replace('_score', '').replace('_', ' ').title())
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


if __name__ == "__main__":
    # Example usage
    # Load your DataFrame here


    df,_ = load_data()
    selected_benchmarks = ['arc_ai2_heb_score', 'copa_heb_score']
    selected_runs = ['qwen8-20-billion', 'Qwen3-8B']

    # Use the fixed function instead of the one from scores_dashboard
    df['model_group'] = df['model_name'].apply(get_base_model_name)

    fig = create_training_progress_plot(df, selected_runs, selected_benchmarks, include_checkpoints=True, subplot_per_benchmark=False)
    fig.show()