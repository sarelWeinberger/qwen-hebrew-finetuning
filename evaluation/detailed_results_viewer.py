import gradio as gr
import pandas as pd
import os
import numpy as np
from typing import Dict, Any, Optional, List

def load_parquet_file(parquet_path: str) -> Optional[pd.DataFrame]:
    """Load a parquet file and return its contents as DataFrame"""
    try:
        df = pd.read_parquet(parquet_path)
        return df
    except Exception as e:
        print(f"Error loading parquet file {parquet_path}: {e}")
        return None

def is_empty(value: Any) -> bool:
    """Check if a value is empty, handling arrays, strings, None, etc."""
    try:
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, str):
            return len(value.strip()) == 0
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        if isinstance(value, np.ndarray):
            return value.size == 0
        if isinstance(value, dict):
            return len(value) == 0
        return False
    except:
        return False

def safe_get_value(obj: Any, key: str, default: Any = '') -> Any:
    """Safely get a value from dict-like object"""
    try:
        if isinstance(obj, dict):
            value = obj.get(key, default)
            if is_empty(value):
                return default
            return value
        return default
    except:
        return default

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that have all empty values"""
    try:
        if df.empty:
            return df
        
        # Check if error or message dataframe
        if 'Error' in df.columns or 'Message' in df.columns:
            return df
        
        cols_to_keep = []
        
        for col in df.columns:
            # Check if column has any non-empty values
            has_data = False
            for val in df[col]:
                if not is_empty(val):
                    has_data = True
                    break
            
            if has_data:
                cols_to_keep.append(col)
        
        # Keep at least one column to avoid empty dataframe
        if len(cols_to_keep) == 0:
            return df
        
        return df[cols_to_keep]
        
    except Exception as e:
        print(f"Error removing empty columns: {e}")
        return df

def extract_simple_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract simple view: doc.query, metric.acc, model_response.output_tokens
    Each row is a sample, columns are the fields
    """
    try:
        simple_records = []
        
        for idx, row in df.iterrows():
            record = {
                'Sample #': idx,
            }
            
            # Extract doc.query
            if 'doc' in df.columns:
                doc = row['doc']
                if isinstance(doc, dict):
                    query = safe_get_value(doc, 'query')
                    if not is_empty(query):
                        # Convert to string and limit length for display
                        record['Query (Input)'] = str(query)[:1000]
                    else:
                        record['Query (Input)'] = ''
                else:
                    record['Query (Input)'] = ''
            else:
                record['Query (Input)'] = ''
            
            # Extract model_response.output_tokens
            if 'model_response' in df.columns:
                model_response = row['model_response']
                if isinstance(model_response, dict):
                    output_tokens = safe_get_value(model_response, 'output_tokens')
                    
                    if not is_empty(output_tokens):
                        # Handle different types of output_tokens
                        if isinstance(output_tokens, (list, tuple, np.ndarray)):
                            # Convert array/list to readable string
                            try:
                                if isinstance(output_tokens, np.ndarray):
                                    if output_tokens.size > 0:
                                        output_str = str(output_tokens.tolist())
                                    else:
                                        output_str = ''
                                else:
                                    output_str = str(output_tokens)
                                record['Model Output'] = output_str[:1000]
                            except:
                                record['Model Output'] = str(output_tokens)[:1000]
                        else:
                            record['Model Output'] = str(output_tokens)[:1000]
                    else:
                        record['Model Output'] = ''
                else:
                    record['Model Output'] = ''
            else:
                record['Model Output'] = ''
            
            # Extract metric.acc
            if 'metric' in df.columns:
                metric = row['metric']
                if isinstance(metric, dict):
                    acc = safe_get_value(metric, 'acc', None)
                    
                    if acc is not None:
                        try:
                            # Handle numpy arrays
                            if isinstance(acc, np.ndarray):
                                if acc.size > 0:
                                    acc = acc.item()
                                else:
                                    acc = None
                            
                            if acc is not None:
                                # Store both numeric and visual representation
                                record['Accuracy'] = float(acc)
                                record['Correct'] = '✓' if (acc == 1 or acc == 1.0 or acc == True) else '✗'
                            else:
                                record['Accuracy'] = None
                                record['Correct'] = '?'
                        except Exception as e:
                            print(f"Error extracting acc: {e}")
                            record['Accuracy'] = None
                            record['Correct'] = '?'
                    else:
                        record['Accuracy'] = None
                        record['Correct'] = '?'
                else:
                    record['Accuracy'] = None
                    record['Correct'] = '?'
            else:
                record['Accuracy'] = None
                record['Correct'] = '?'
            
            simple_records.append(record)
        
        if not simple_records:
            return pd.DataFrame({'Message': ['No data available in expected format']})
        
        result_df = pd.DataFrame(simple_records)
        
        # Remove columns with all empty values
        result_df = remove_empty_columns(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"Error in extract_simple_view: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'Error': [f'Failed to create simple view: {str(e)}']})

def safe_convert_to_string(obj: Any, max_length: int = 1000) -> str:
    """Safely convert any object to string representation"""
    try:
        if is_empty(obj):
            return ''
        elif isinstance(obj, np.ndarray):
            # Handle numpy arrays
            if obj.size == 0:
                return ''
            elif obj.size == 1:
                return str(obj.item())
            else:
                result = str(obj.tolist())
                if len(result) > max_length:
                    return result[:max_length] + '...'
                return result
        elif isinstance(obj, (dict, list, tuple)):
            import json
            try:
                result = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
                if len(result) > max_length:
                    return result[:max_length] + '...'
                return result
            except:
                result = str(obj)
                if len(result) > max_length:
                    return result[:max_length] + '...'
                return result
        else:
            result = str(obj)
            if len(result) > max_length:
                return result[:max_length] + '...'
            return result
    except Exception as e:
        return f"<Error converting: {str(e)}>"

def extract_full_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract full view: show entire parquet as dataframe
    Each row is a sample, columns are all available fields
    """
    try:
        # Create a flattened view of the dataframe
        full_records = []
        
        for idx, row in df.iterrows():
            record = {
                'Sample #': idx,
            }
            
            # Add all columns from the original dataframe
            for col in df.columns:
                value = row[col]
                
                # Handle nested structures
                if isinstance(value, dict):
                    # Flatten dictionary fields
                    for key, val in value.items():
                        full_col_name = f"{col}.{key}"
                        record[full_col_name] = safe_convert_to_string(val)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    # Convert lists/tuples/arrays to string
                    record[col] = safe_convert_to_string(value)
                else:
                    record[col] = safe_convert_to_string(value)
            
            full_records.append(record)
        
        if not full_records:
            return pd.DataFrame({'Message': ['No data available']})
        
        result_df = pd.DataFrame(full_records)
        
        # Remove columns with all empty values
        result_df = remove_empty_columns(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"Error in extract_full_view: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'Error': [f'Failed to create full view: {str(e)}']})

def create_detailed_viewer(parquet_path: str, view_mode: str = "Simple") -> pd.DataFrame:
    """Create detailed view based on parquet file"""
    if not os.path.exists(parquet_path):
        return pd.DataFrame({'Error': [f'File not found: {parquet_path}']})
    
    df = load_parquet_file(parquet_path)
    if df is None or df.empty:
        return pd.DataFrame({'Error': ['Failed to load parquet file or file is empty']})
    
    if view_mode == "Simple":
        return extract_simple_view(df)
    else:  # Full view
        return extract_full_view(df)

def launch_detailed_viewer(model_name: str, dataset: str, timestamp: str, 
                          benchmark_results_dir: str = "/home/ubuntu/qwen-hebrew-finetuning/evaluation"):
    """Launch a detailed viewer for a specific result"""
    
    # Construct parquet file path
    parquet_filename = f"{dataset}.parquet"
    parquet_path = os.path.join(benchmark_results_dir, model_name, timestamp, parquet_filename)
    
    def update_view(view_mode):
        return create_detailed_viewer(parquet_path, view_mode)
    
    with gr.Blocks(title=f"Results: {dataset}") as viewer:
        gr.Markdown(f"# 📊 Detailed Results Viewer")
        gr.Markdown(f"**Model:** {model_name}")
        gr.Markdown(f"**Dataset:** {dataset}")
        gr.Markdown(f"**Timestamp:** {timestamp}")
        gr.Markdown(f"**File:** `{parquet_path}`")
        
        view_selector = gr.Radio(
            choices=["Simple", "Full"],
            value="Simple",
            label="View Mode",
            info="Simple: Shows doc.query, metric.acc, and model_response.output_tokens | Full: Shows all columns from parquet file"
        )
        
        results_table = gr.Dataframe(
            value=create_detailed_viewer(parquet_path, "Simple"),
            wrap=True,
            max_height=600,
            interactive=False,
            line_breaks=True
        )
        
        # Add statistics
        def get_stats(view_mode):
            df_view = create_detailed_viewer(parquet_path, view_mode)
            if 'Error' in df_view.columns or 'Message' in df_view.columns:
                return "No statistics available"
            
            stats = f"**Total Samples:** {len(df_view)}\n\n"
            stats += f"**Columns Displayed:** {len(df_view.columns)}\n\n"
            
            if 'Correct' in df_view.columns:
                correct_count = (df_view['Correct'] == '✓').sum()
                total_count = len(df_view)
                accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
                stats += f"**Correct:** {correct_count}/{total_count} ({accuracy:.2f}%)\n\n"
            
            if 'Accuracy' in df_view.columns:
                avg_acc = df_view['Accuracy'].mean()
                if not pd.isna(avg_acc):
                    stats += f"**Average Accuracy:** {avg_acc:.4f}\n\n"
            
            return stats
        
        stats_box = gr.Markdown(value=get_stats("Simple"))
        
        view_selector.change(
            fn=lambda vm: (update_view(vm), get_stats(vm)),
            inputs=[view_selector],
            outputs=[results_table, stats_box]
        )
        
        refresh_btn = gr.Button("🔄 Refresh View", variant="secondary")
        refresh_btn.click(
            fn=lambda vm: (update_view(vm), get_stats(vm)),
            inputs=[view_selector],
            outputs=[results_table, stats_box]
        )
    
    return viewer

# Standalone function for external launch
def create_viewer_interface(parquet_path: str) -> gr.Blocks:
    """Create a standalone viewer interface for a parquet file"""
    
    def update_view(view_mode):
        return create_detailed_viewer(parquet_path, view_mode)
    
    def get_stats(view_mode):
        df_view = create_detailed_viewer(parquet_path, view_mode)
        if 'Error' in df_view.columns or 'Message' in df_view.columns:
            return "No statistics available"
        
        stats = f"**Total Samples:** {len(df_view)}\n\n"
        stats += f"**Columns Displayed:** {len(df_view.columns)}\n\n"
        
        if 'Correct' in df_view.columns:
            correct_count = (df_view['Correct'] == '✓').sum()
            total_count = len(df_view)
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            stats += f"**Correct:** {correct_count}/{total_count} ({accuracy:.2f}%)\n\n"
        
        if 'Accuracy' in df_view.columns:
            avg_acc = df_view['Accuracy'].mean()
            if not pd.isna(avg_acc):
                stats += f"**Average Accuracy:** {avg_acc:.4f}\n\n"
        
        return stats
    
    with gr.Blocks(title="Detailed Results Viewer") as viewer:
        gr.Markdown("# 📊 Detailed Results Viewer")
        gr.Markdown(f"**File:** `{parquet_path}`")
        
        view_selector = gr.Radio(
            choices=["Simple", "Full"],
            value="Simple",
            label="View Mode",
            info="Simple: Shows doc.query, metric.acc, and model_response.output_tokens | Full: Shows all columns from parquet file"
        )
        
        stats_box = gr.Markdown(value=get_stats("Simple"))
        
        results_table = gr.Dataframe(
            value=create_detailed_viewer(parquet_path, "Simple"),
            wrap=True,
            max_height=600,
            interactive=False,
            line_breaks=True
        )
        
        view_selector.change(
            fn=lambda vm: (update_view(vm), get_stats(vm)),
            inputs=[view_selector],
            outputs=[results_table, stats_box]
        )
        
        refresh_btn = gr.Button("🔄 Refresh View", variant="secondary")
        refresh_btn.click(
            fn=lambda vm: (update_view(vm), get_stats(vm)),
            inputs=[view_selector],
            outputs=[results_table, stats_box]
        )
        
        # Add download button - FIXED
        download_btn = gr.Button("💾 Download Current View as CSV", variant="secondary")
        
        def download_csv(view_mode):
            """Generate CSV file and return the file path"""
            try:
                df = create_detailed_viewer(parquet_path, view_mode)
                
                # Generate unique filename with timestamp
                import time
                timestamp = int(time.time())
                filename = f"detailed_view_{timestamp}.csv"
                temp_path = os.path.join("/tmp", filename)
                
                # Save to CSV
                df.to_csv(temp_path, index=False, encoding='utf-8')
                
                print(f"CSV saved to: {temp_path}")
                return temp_path
            except Exception as e:
                print(f"Error creating CSV: {e}")
                import traceback
                traceback.print_exc()
                # Return error file
                error_path = "/tmp/error.txt"
                with open(error_path, 'w') as f:
                    f.write(f"Error creating CSV: {str(e)}")
                return error_path
        
        # Use gr.File for download
        download_output = gr.File(label="Download CSV", visible=True)
        
        download_btn.click(
            fn=download_csv,
            inputs=[view_selector],
            outputs=download_output
        )
    
    return viewer

if __name__ == "__main__":
    # Example usage
    parquet_path = "/home/ec2-user/test_output/hellaswag_heb/scores_sum/2025-10-15T21-51-38/details/home/ec2-user/models/Qwen3-14B/2025-10-15T21-52-23.466806/details_community|hellaswag_heb|3_2025-10-15T21-52-23.466806.parquet"
    viewer = create_viewer_interface(parquet_path)
    viewer.launch(share=True)  # Set share=True to get a public link