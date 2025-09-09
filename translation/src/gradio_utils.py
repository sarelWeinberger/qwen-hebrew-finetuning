import gradio as gr
import pandas as pd
import os
from time import gmtime

"""
This code is (mostly) brought to you by Gemini.
"""

# --- Configuration & Global Data ---

CHOICE1_OPTIONS = ["", "1", "2", "BOTH", "SKIP"]
CATEGORY_OPTIONS_BASE = [
    "Adequacy - Mistranslation", "Adequacy - Omission", "Adequacy - Addition",
    "Adequacy - TerminologyNamedEntity", "Adequacy - CulturalReference",
    "Fluency - Agreement", "Fluency - MorphologyFunction", "Fluency - WordOrderSyntax",
    "Fluency - OrthographyPunct", "LocaleStyle - Register", "LocaleStyle - Conventions",
]
CATEGORY_OPTIONS = [""] + CATEGORY_OPTIONS_BASE
SEVERITY_OPTIONS_BASE = ["minor", "major", "critical"]
SEVERITY_OPTIONS = [""] + SEVERITY_OPTIONS_BASE
NUM_MQM_PAIRS = 6 # Total number of Category/Severity pairs


# --- Helper Functions for Data Handling ---

def initialize_annotation_columns(df):
    """Ensures all required annotation columns exist in the DataFrame."""
    if 'rating' not in df.columns:
        df['rating'] = ""
    for i in range(NUM_MQM_PAIRS):
        if f'category_annotation_{i+1}' not in df.columns:
            df[f'category_annotation_{i+1}'] = ""
        if f'severity_annotation_{i+1}' not in df.columns:
            df[f'severity_annotation_{i+1}'] = ""
    if 'gold' not in df.columns:
        df['gold'] = ""
    return df

def load_data(base_df, save_filename):
    """Loads the initial DataFrame, merging existing annotations if a save file is found."""
    if os.path.exists(save_filename):
        # print(f"Found existing annotations file: {save_filename}. Loading...")
        try:
            loaded_df = pd.read_csv(save_filename) if save_filename.endswith(".csv") else pd.read_json(save_filename, orient='records')
            loaded_df = loaded_df.fillna('')
            loaded_df['rating'] = loaded_df['rating'].apply(str).replace('1.0', '1').replace('2.0', '2')
            
            # Ensure all columns from loaded_df are in base_df before merge
            for col in loaded_df.columns:
                if col not in base_df.columns:
                    base_df[col] = "" # Add missing columns from loaded_df to base_df

            # Assuming 'text_column' can uniquely identify rows for merging. A dedicated ID is safer.
            # Using a temporary ID column for merging if 'text_column' isn't suitable or missing
            if 'text_column' in base_df.columns and 'text_column' in loaded_df.columns:
                # Create a temporary unique ID for robust merging if 'text_column' alone isn't unique
                if not base_df['text_column'].is_unique:
                    assert False, "'text_column' is not unique in the given dataframe."
                    return None
                else: # If text_column is unique, use it directly for update
                    base_df = base_df.set_index('text_column')
                    base_df.update(loaded_df.set_index('text_column'))
                    base_df = base_df.reset_index()
                # print("Successfully merged existing annotations.")
            else:
                print("Warning: Could not merge annotations. 'text_column' not found or not suitable as a key. Returning initial DataFrame with existing columns.")
                # If merging isn't possible, just return loaded_df if it's considered the source of truth,
                # or base_df with initialized columns. For annotation, loaded_df is usually preferred.
                return initialize_annotation_columns(loaded_df)
        except Exception as e:
            print(f"Error loading or merging annotations: {e}")
            return initialize_annotation_columns(base_df)
            
    return initialize_annotation_columns(base_df)

def save_dataframe(df, save_filename):
    """Saves the DataFrame and returns a status message for Gradio's notification system."""
    time_files_lst = os.listdir('labeled_files/time_checkup')
    if len(time_files_lst) > 150:
        for f in time_files_lst:
            if os.path.isfile(os.path.join('labeled_files/time_checkup', f)):
                os.remove(os.path.join('labeled_files/time_checkup', f))
    try:
        if save_filename.endswith(".csv"):
            df.to_csv(save_filename, index=False, encoding='utf-8')
            current_time = gmtime()
            backup_filename = f'labeled_files/time_checkup/backup_{df.shape[0]}_{current_time.tm_mday}_{current_time.tm_mon}_{current_time.tm_hour}{current_time.tm_min}.csv'
            df.to_csv(backup_filename, index=False, encoding='utf-8')
        else:
            df.to_json(save_filename, orient='records', indent=4, force_ascii=False)
        return gr.Info(f"Annotations saved to {save_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return gr.Warning(f"Could not save annotations! Error: {e}")

# --- Gradio Core Functions ---

# New function to be called on demo.load to get the initial state including loading the latest data
def get_initial_state(initial_df_placeholder, save_filename):
    """
    Loads the latest data from the save_filename and initializes the UI.
    This function is called on demo.load.
    """
    # Use the initial_df_placeholder to get the structure if save_filename doesn't exist yet
    # Otherwise, load from save_filename which contains the most up-to-date annotations.
    loaded_data = load_data(initial_df_placeholder.copy() if initial_df_placeholder is not None else pd.DataFrame(), save_filename)
    initial_index = 0
    ui_values = get_sample_data(loaded_data, initial_index)
    return [loaded_data, initial_index] + ui_values


def save_current_and_get_new_state(df, index, direction, jump_to_str, save_filename, rating, gold_text, *mqm_annotations):
    """Saves current data, calculates the next index, and returns the full new UI state."""
    # 1. Save current annotations to the DataFrame
    df.loc[index, 'rating'] = rating if rating else ""
    df.loc[index, 'gold'] = gold_text
    
    for i in range(NUM_MQM_PAIRS):
        cat_val = mqm_annotations[i * 2]
        sev_val = mqm_annotations[i * 2 + 1]
        df.loc[index, f'category_annotation_{i+1}'] = cat_val if cat_val else ""
        df.loc[index, f'severity_annotation_{i+1}'] = sev_val if sev_val else ""

    # 2. Save the DataFrame to disk
    save_dataframe(df, save_filename)

    # 3. Calculate the new index
    new_index = index
    if direction == "next":
        if new_index < len(df) - 1: new_index += 1
        else: gr.Info("You are at the last sample.")
    elif direction == "prev":
        if new_index > 0: new_index -= 1
        else: gr.Info("You are at the first sample.")
    elif direction == "jump":
        try:
            target_index = int(jump_to_str) - 1
            if 0 <= target_index < len(df): new_index = target_index
            else: gr.Warning(f"Invalid index. Please enter a number between 1 and {len(df)}.")
        except (ValueError, TypeError):
            gr.Warning("Please enter a valid number to jump to an index.")
            
    # 4. Load data for the new index
    new_ui_values = get_sample_data(df, new_index)
    
    # 5. Return all new values for the UI components
    return [df, new_index] + new_ui_values

def get_sample_data(df, index):
    """Retrieves and formats data for a given index to populate the UI."""
    if not 0 <= index < len(df):
        error_label = f"Jump to Index (Invalid: {index})"
        empty_mqm = [""] * (NUM_MQM_PAIRS * 2)
        # Return empty values for all fields, including the two new text boxes
        return ["", "", "", "", gr.update(label=error_label, value=None)] + empty_mqm

    row = df.iloc[index]
    
    def get_val(col_name, default=""):
        val = row.get(col_name)
        return default if pd.isna(val) else val

    # CHANGE: Fetch text for both original and new text boxes
    original_text_to_display = get_val('text_column')
    new_text_to_display = get_val('new_text_column') # Assumes this column exists
    
    rating = get_val('rating')
    gold_text = get_val('gold')
    
    jump_label = f"Jump to Index (Current: {index + 1} of {len(df)})"
    jump_box_update = gr.update(label=jump_label, value=None)
    
    mqm_values = []
    for i in range(NUM_MQM_PAIRS):
        mqm_values.append(get_val(f'category_annotation_{i+1}'))
        mqm_values.append(get_val(f'severity_annotation_{i+1}'))
        
    # CHANGE: Return the two text values at the beginning of the list
    return [original_text_to_display, new_text_to_display, rating, gold_text, jump_box_update] + mqm_values

def run_annotator(dataframe, save_filename="annotated_data.csv", extra_title=''):
    """Main function to configure and launch the Gradio annotator interface."""
    if not dataframe['text_column'].is_unique:
        print("the 'text_column' column is not unique! This can't be happening!!")
        return None
    
    # initial_df is now just a placeholder for the structure, it won't be used directly for UI state
    # The actual initial loading for UI will happen in get_initial_state
    initial_df_structure = initialize_annotation_columns(dataframe.copy()) # Only to get the column structure

    # CHANGE: Added a CSS rule for the RTL textbox
    custom_css = """
    .gradio-container { font-family: Calibri, sans-serif !important; }
    #new-text-rtl textarea { direction: rtl; text-align: right; }

    #mqm-accordion-1 { background-color: #e7f5ff !important; }
    #mqm-accordion-2 { background-color: #e8f5e9 !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Data Annotator", css=custom_css) as demo:
        state_df = gr.State(value=None) # Initialize with None, it will be loaded by demo.load
        state_index = gr.State(value=0)
        
        gr.Markdown("# Text Annotator" + " " + extra_title)
        
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous Sample")
            next_btn = gr.Button("Next Sample ➡️")
            jump_index_num = gr.Number(label="Jump to Index", precision=0)
            jump_btn = gr.Button("Go")
        
        gr.Markdown("---")
        
        # CHANGE: Split the text display into two boxes
        original_text_display = gr.Textbox(label="Original", lines=4, interactive=False)
        # The elem_id is used by the CSS to apply RTL styling
        new_text_display = gr.Textbox(label="New", lines=4, interactive=False, elem_id="new-text-rtl")

        with gr.Group():
            gr.Markdown("### General Annotation")
            rating_dd = gr.Dropdown(CHOICE1_OPTIONS, label="Rating")
            gold_text = gr.Textbox(label="Gold (Corrected Text)", lines=6)

        cat_dds, sev_dds = [], []
        with gr.Accordion("MQM - Option 1", open=True, elem_id="mqm-accordion-1"):
            with gr.Row():
                for i in range(3):
                    with gr.Column():
                        cat = gr.Dropdown(CATEGORY_OPTIONS, label="")
                        sev = gr.Dropdown(SEVERITY_OPTIONS, label="")
                        cat_dds.append(cat)
                        sev_dds.append(sev)

        with gr.Accordion("MQM - Option 2", open=True, elem_id="mqm-accordion-2"):
            with gr.Row():
                for i in range(3, 6):
                    with gr.Column():
                        cat = gr.Dropdown(CATEGORY_OPTIONS, label="")
                        sev = gr.Dropdown(SEVERITY_OPTIONS, label="")
                        cat_dds.append(cat)
                        sev_dds.append(sev)
        
        mqm_dropdowns = [item for pair in zip(cat_dds, sev_dds) for item in pair]
        annotation_inputs = [rating_dd, gold_text] + mqm_dropdowns
        
        # CHANGE: The ui_outputs list now includes the two new textboxes
        ui_outputs = [original_text_display, new_text_display, rating_dd, gold_text, jump_index_num] + mqm_dropdowns

        # Event Handling
        # CHANGE: Call get_initial_state on load to re-load the data from disk
        demo.load(
            fn=get_initial_state,
            inputs=[gr.State(initial_df_structure), gr.State(save_filename)], # Pass a placeholder for initial structure and save_filename
            outputs=[state_df, state_index] + ui_outputs
        )

        next_btn.click(
            fn=save_current_and_get_new_state,
            inputs=[state_df, state_index, gr.State("next"), gr.State(None), gr.State(save_filename)] + annotation_inputs,
            outputs=[state_df, state_index] + ui_outputs
        )
        prev_btn.click(
            fn=save_current_and_get_new_state,
            inputs=[state_df, state_index, gr.State("prev"), gr.State(None), gr.State(save_filename)] + annotation_inputs,
            outputs=[state_df, state_index] + ui_outputs
        )
        jump_btn.click(
            fn=save_current_and_get_new_state,
            inputs=[state_df, state_index, gr.State("jump"), jump_index_num, gr.State(save_filename)] + annotation_inputs,
            outputs=[state_df, state_index] + ui_outputs
        )

    demo.launch(share=True, inline=False)

    return demo


def check():
    # CHANGE: Added 'new_text_column' to the example data
    example_data = {
        'text_column': [
            "This is the first sample text. It contains some information that needs to be categorized.",
            "Here is the second sample, a bit longer, detailing a different subject.",
            "The third sample focuses on a specific topic with unique characteristics.",
        ],
        'new_text_column': [
            "טקסט חדש לדוגמה, שצריך להיות מוצג מימין לשמאל.",
            "זוהי הדוגמה השנייה, והיא גם כתובה בעברית.",
            "והדוגמה השלישית והאחרונה לבדיקה.",
        ],
        'gold': [
            'Gold 1',
            'Gold 2',
            'Gold 3',
        ],
        'original_id': [101, 102, 103],
        'source': ['email', 'web', 'report']
    }
    my_dataframe = pd.DataFrame(example_data)
    # Initialize the 'gold' column so it exists for the app
    my_dataframe['gold'] = [
        "This is the first sample text, which has information needing categorization.",
        "This is the second, slightly longer sample, which details a different subject.",
        "The third sample is focused on a specific topic that has unique characteristics.",
    ]
    
    my_annotation_output_file = "my_project_annotations_gradio.csv" 
    extra_title = ' - Check data'
    run_annotator(my_dataframe, save_filename=my_annotation_output_file, extra_title=extra_title)
    print(f"\nAnnotation process finished. Check the output file: '{my_annotation_output_file}'")
