"""
This file contains functions to plot the results of the experts distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

experts_columns = [f'Expert_{i}' for i in range(128)]


def into_top(arr, top_1, top_half):
    top_indices = np.argsort(arr, axis=1)[:, -top_1:]
    top_half_indices = np.argsort(arr, axis=1)[:, -top_half:]
    arr = np.zeros_like(arr)
    row_indices = np.arange(arr.shape[0])[:, np.newaxis]
    arr[row_indices, top_half_indices] = 0.3
    arr[row_indices, top_indices] = 1

    return arr


def plot_heatmap_experts(df, vmax=1, figsize=(15, 5), showdiff=False, show_top=False, from_layer=0, to_layer=48):    
    for lan in ['English', 'Hebrew']:
        use = df[df['Language'] == lan]
        use = use.set_index('Layer')[experts_columns]

        if show_top:
            use = into_top(use, 3, 8)
            vmax=1
        
        plt.figure(figsize=figsize)
        plt.imshow(use, cmap='Oranges', origin='lower', vmin=0, vmax=vmax, aspect='auto')
        plt.yticks(range(use.shape[0]), range(1, use.shape[0] + 1))
        plt.title(f'{lan} layers {from_layer} - {to_layer}')
        plt.colorbar()
        plt.show()

    if showdiff:
        values_1 = df[df['Language'] == 'English'][experts_columns].values
        values_2 = df[df['Language'] == 'Hebrew'][experts_columns].values
        if show_top:
            values_1 = into_top(values_1, 3, 8)
            values_2 = into_top(values_2, 3, 8)
        diff = np.abs(values_1 - values_2)
        fill = diff * 0 + 1
    
        # plt.figure(figsize=figsize)
        # plt.imshow(diff, cmap='Oranges', origin='lower', vmin=0, vmax=vmax, aspect='auto')
        # plt.yticks(range(use.shape[0]), range(1, use.shape[0] + 1))

        show_dist = np.stack([values_1, values_2, diff, fill], axis=0).transpose(1, 0, 2).reshape(-1, values_1.shape[1])
        plt.figure(figsize=(figsize[0], figsize[1] * 3))
        plt.imshow(show_dist, cmap='Oranges', origin='lower', vmin=0, vmax=vmax, aspect='auto')
        plt.yticks(range(0, values_1.shape[0] * 4, 4), range(1, values_1.shape[0] + 1))
        
        plt.title(f'diff layers {from_layer} - {to_layer}')
        plt.colorbar()
        plt.show()


def comp_one_lan(df_lst, vmax=1, lan='Hebrew', figsize=(15, 5), show_top=False, from_layer=0, to_layer=48):    
    values_lst = [df[df['Language'] == lan][experts_columns].values for df in df_lst]
    if show_top:
        vmax = 1
        values_lst = [into_top(v, 3, 8) for v in values_lst]
        
    fill = values_lst[0] * 0 + 1

    # plt.figure(figsize=figsize)
    # plt.imshow(diff, cmap='Oranges', origin='lower', vmin=0, vmax=vmax, aspect='auto')
    # plt.yticks(range(use.shape[0]), range(1, use.shape[0] + 1))

    show_dist = np.stack(values_lst + [fill], axis=0).transpose(1, 0, 2).reshape(-1, fill.shape[1])
    plt.figure(figsize=(figsize[0], figsize[1] * (1 + len(values_lst))))
    plt.imshow(show_dist, cmap='Oranges', origin='lower', vmin=0, vmax=vmax, aspect='auto')
    plt.yticks(range(0, fill.shape[0] * (1 + len(values_lst)), (1 + len(values_lst))), range(from_layer, to_layer))
    
    plt.title(f'{lan} - stacked datasets layers {from_layer} - {to_layer}')
    plt.colorbar()
    plt.show()


map_from_3 = {
    0.0: 0,
    0.2: 1,
    0.5: 2,
    1.0: 3,
}
map_from_8 = {
    0.0: 0,
    0.067: 1,
    0.143: 2,
    0.231: 3,
    0.333: 4,
    0.455: 5,
    0.6: 6,
    0.778: 7,
    1.0: 8,
}


def plot_moe_analysis_from_log(log_file_path: str, label='', axs=None):
    """
    Reads a MoE analysis log file, calculates the average of key metrics
    across all samples per layer, and plots the results.

    Args:
        log_file_path (str): The full path to the .log file.
    """
    # Dictionary to store the collected data from the log file.
    # Each key will hold a list of lists, where each inner list represents
    # the 48 layer values for a single sample (prompt).
    data = {
        'Cosine_Dist': [],
        'Overlap_3': [],
        'Overlap_8': []
    }

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    in_progression_block = False
    current_sample_data = []

    # Iterate through each line of the file to parse the data
    for line in lines:
        # A new "LAYER PROGRESSION" block indicates a new sample
        if "LAYER PROGRESSION:" in line:
            # If we have data from the previous block, process and store it
            if current_sample_data:
                # Transpose the list of tuples to get lists of values for each metric
                cos_dist_vals, o3_vals, o8_vals = zip(*current_sample_data)
                data['Cosine_Dist'].append(list(cos_dist_vals))
                data['Overlap_3'].append(list(o3_vals))
                data['Overlap_8'].append(list(o8_vals))
            
            # Reset for the new sample block
            in_progression_block = True
            current_sample_data = []
            continue

        # The "AGGREGATED" section marks the end of all progression blocks
        if "AGGREGATED AVERAGE DISTRIBUTIONS" in line:
            # Process and store the very last sample's data before stopping
            if current_sample_data:
                cos_dist_vals, o3_vals, o8_vals = zip(*current_sample_data)
                data['Cosine_Dist'].append(list(cos_dist_vals))
                data['Overlap_3'].append(list(o3_vals))
                data['Overlap_8'].append(list(o8_vals))
            break  # Stop parsing, as we have all the data we need

        # If we are inside a progression block, look for data lines
        if in_progression_block:
            parts = line.split()
            # A valid data line has several parts and the 4th element is the layer number
            if len(parts) > 4 and parts[3].isdigit():
                try:
                    # Extract the metrics from the end of the line for robustness
                    cosine_dist = float(parts[-1])
                    overlap_8 = map_from_8[np.round(float(parts[-2]), 3)]
                    overlap_3 = map_from_3[float(parts[-3])]
                    current_sample_data.append((cosine_dist, overlap_3, overlap_8))
                except (ValueError, IndexError):
                    # This line looked like a data line but wasn't, so we skip it
                    continue

    # After parsing, check if any data was actually collected
    if not data['Cosine_Dist']:
        print("No valid 'LAYER PROGRESSION' data could be extracted from the log file.")
        return

    # --- Data Aggregation and Plotting ---

    # Calculate the average across all samples for each layer using numpy
    try:
        avg_cosine_dist = np.mean(data['Cosine_Dist'], axis=0)
        avg_overlap_3 = np.mean(data['Overlap_3'], axis=0)
        avg_overlap_8 = np.mean(data['Overlap_8'], axis=0)
    except Exception as e:
        print(f"An error occurred during data aggregation. Ensure all samples have 48 layers. Error: {e}")
        return

    num_layers = len(avg_cosine_dist)
    layers = range(num_layers)
    num_samples = len(data['Cosine_Dist'])

    # plt.style.use('seaborn-v0_8-whitegrid')
    if axs is None:
        # Create the plot
        plt.figure(figsize=(14, 8))
    
        plt.plot(layers, avg_cosine_dist, marker='o', linestyle='-', label=f'Average Cosine Distance')
        plt.plot(layers, avg_overlap_3, marker='s', linestyle='--', label=f'Average Overlap 3')
        plt.plot(layers, avg_overlap_8, marker='^', linestyle=':', label=f'Average Overlap 8')
    
        # Adding plot titles and labels
        plt.title(f'{label} - Average MoE Metrics per Layer (#{num_samples} samples)', fontsize=16)
        plt.xlabel('Layer Number', fontsize=12)
        plt.ylabel('Average Value', fontsize=12)
        plt.xticks(np.arange(0, num_layers, 2))  # Set x-axis ticks to be every 2 layers
        plt.legend(fontsize=10)
        plt.tight_layout()  # Adjust layout to make room for labels
    
        # Display the plot
        plt.show()
    else:
        axs[0].plot(layers, avg_cosine_dist, marker='o', linestyle='-', label=f'{label} - Average Cosine Distance')
        axs[1].plot(layers, avg_overlap_3, marker='s', linestyle='-', label=f'{label} - Average Overlap 3')
        axs[2].plot(layers, avg_overlap_8, marker='^', linestyle='-', label=f'{label} - Average Overlap 8')
        
    