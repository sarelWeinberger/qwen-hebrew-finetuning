import os
import re
import string
import collections
import json
import numpy as np
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import  Metrics
# from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric as loglikelihood_acc_metric
# from lighteval.metrics.normalizations import LogProbTokenNorm
# from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
# from lighteval.metrics.utils.metric_utils import (
#     CorpusLevelMetric,
#     CorpusLevelMetricGrouping,
#     Metric,
#     MetricCategory,
#     MetricGrouping,
#     MetricUseCase,
#     SampleLevelMetric,
#     SampleLevelMetricGrouping,
# )

# ============================================================
# Custom HEQ Evaluation Functions (Shaltiel Shmidman version)
# ============================================================

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
HEB_BENCHMARKS_DIR_PATH = os.getenv("HEB_BENCHMARKS_DIR_PATH", "/home/ec2-user/qwen-hebrew-finetuning/evaluation/heb_bnch")
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
INTEGER_INDICES = list(map(str, list(range(1, 27))))

def normalize_answer(s):
    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(s.replace("<pad>", "").replace("</s>", "").strip())
            )
        )
    )


def extract_final_answer(pred) -> str:
    """
    Robustly extract the part of the answer after '####' if it exists.
    Works with strings, ModelResponse objects, or arbitrarily nested lists of either.
    """

    # DEBUG: Print raw input
    print(f"DEBUG extract_final_answer input type: {type(pred)}")

    # Recursively flatten until we get a string or ModelResponse
    def flatten(p):
        while isinstance(p, list):
            if not p:
                return ""
            p = p[0]  # Keep unwrapping lists
        return p

    pred = flatten(pred)
    print(f"DEBUG after flatten type: {type(pred)}")

    # If it's a ModelResponse object, extract text
    if hasattr(pred, "text_post_processed") and pred.text_post_processed:
        pred_text = pred.text_post_processed
        print(f"DEBUG using text_post_processed: {repr(pred_text)}")
        # Extract from list if needed
        if isinstance(pred_text, list):
            pred_text = pred_text[0] if pred_text else ""
    elif hasattr(pred, "text") and pred.text:
        pred_text = pred.text
        print(f"DEBUG using text: {repr(pred_text)}")
        # Extract from list if needed
        if isinstance(pred_text, list):
            pred_text = pred_text[0] if pred_text else ""
    else:
        pred_text = str(pred)
        print(f"DEBUG converted to string: {repr(pred_text)}")

    # Ensure we have a string at this point
    if not isinstance(pred_text, str):
        pred_text = str(pred_text)

    print(f"DEBUG raw text before processing: {repr(pred_text)}")

    # Remove special tokens and stop sequences
    pred_text = pred_text.replace("<pad>", "").replace("</s>", "").replace("[ANSWER_END]", "").strip()
    
    # Keep only the part after '####' if it exists
    if "####" in pred_text:
        pred_text = pred_text.split("####")[-1].strip()
        print(f"DEBUG after #### split: {repr(pred_text)}")
    else:
        print(f"DEBUG no #### found, looking for numbers in: {repr(pred_text)}")
        # If no ####, try to extract number from the text
        import re
        
        # First try to find a standalone number at the end
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\s*$', pred_text)
        if number_match:
            pred_text = number_match.group(1)
            print(f"DEBUG extracted number from end: {repr(pred_text)}")
        else:
            # Look for numbers in calculation format like "27+4-3=28"
            calc_match = re.search(r'=\s*(\d+(?:\.\d+)?)\s*$', pred_text)
            if calc_match:
                pred_text = calc_match.group(1)
                print(f"DEBUG extracted number from calculation: {repr(pred_text)}")
            else:
                # Try to find the last number in the text
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', pred_text)
                if numbers:
                    pred_text = numbers[-1]
                    print(f"DEBUG using last number found: {repr(pred_text)}")
                else:
                    print(f"DEBUG no numbers found, keeping full text")

    # Normalize
    result = normalize_answer(pred_text)
    print(f"DEBUG final normalized: {repr(result)}")
    
    return result


def gsm8k_final_acc_eval_fn(**kwargs):
    # Lighteval passes:
    # model_response=model_response, doc=doc (formatted_doc), etc.
    model_response = kwargs.get("model_response")
    formatted_doc = kwargs.get("doc")  # optional
    
    if not model_response:
        print("DEBUG: No model response")
        return 0.0

    # model_response can be a list of strings
    if isinstance(model_response, list):
        pred = extract_final_answer(model_response[0])
    else:
        pred = extract_final_answer(model_response)

    # Extract golds from formatted_doc
    golds = []
    if formatted_doc and hasattr(formatted_doc, "choices") and formatted_doc.choices:
        for choice in formatted_doc.choices:
            # remove the prefix "תשובה: " if present
            text = choice
            if text.startswith("תשובה:"):
                text = text[len("תשובה:"):].strip()
            # Also remove [ANSWER_END] from gold answers
            text = text.replace("[ANSWER_END]", "").strip()
            golds.append(text)

    print(f"DEBUG RAW GOLD CHOICES: {formatted_doc.choices if formatted_doc and hasattr(formatted_doc, 'choices') else 'None'}")
    print(f"DEBUG PROCESSED GOLDS: {golds}")

    golds_norm = [extract_final_answer(x) for x in golds if x]

    # DEBUG: Print the comparison
    print(f"PRED ANSWER -> '{pred}'")
    print(f"REAL ANSWER -> {golds_norm}")
    
    # Check for exact match
    exact_match = pred in golds_norm
    
    print(f"EXACT MATCH: {exact_match}")
    print("=" * 70)

    return float(exact_match)

FEW_SHOT_PATH_gsm8k = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/gsm8k_heb/train.jsonl"

few_shots_list = []
with open(FEW_SHOT_PATH_gsm8k, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        few_shots_list.append({"query": data["query"], "gold": data["gold"]})


# ============================================================
# Wrap HEQ as a Lighteval Metric
# ============================================================
# need to chHANGE TO CorpusLevelMetric FOR NEW VERSION OF LIGHTEVAL
# GSM8KFinalAccMetric = Metric( 
#     metric_name="gsm8k_final_acc",
#     higher_is_better=True,
#     category=MetricCategory.GENERATIVE,
#     use_case=MetricUseCase.ACCURACY,
#     sample_level_fn=gsm8k_final_acc_eval_fn,
#     corpus_level_fn=np.mean,
# )
# ============================================================
# Prompt functions
# ============================================================

printed_prompt = False  # global flag

def gsm8k_heb_fewshot_prompt(line, task_name="gsm8k_heb"):
    global printed_prompt
    
    few_shot_text = ""
    for i, shot in enumerate(few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        ans = shot["gold"].replace('\n', ' ').strip()
        few_shot_text += f"שאלה {i+1}: {qs}\nפתור שלב אחר שלב ותן את התשובה הסופית: {ans}[ANSWER_END]\n\n"

    target_q = line["query"].replace('\n', ' ').strip()
    full_prompt = few_shot_text + f"על סמך השאלות שניתנו פתור את בעיית המתמטיקה הבאה שלב אחר שלב וספק את התשובה הסופית: {target_q}"

    if not printed_prompt:
        print("="*50)
        print("FULL PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_prompt = True

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        gold_index=0,
        choices=['תשובה: ' + line["gold"].replace('\n', ' ').strip() + '[ANSWER_END]']
    )


# ============================================================
# ARC 
# ============================================================
ARC_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/arc_ai2_heb/train.jsonl"

# arc_few_shots_list = []

# with open(ARC_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         if i >= 5:  # take only 25 few-shots
#             break
#         data = json.loads(line)
#         choices = data.get("choices", [])
#         answer_index = data.get("answer_index", 0)  # default 0 if missing
#         gold_choice = choices[answer_index] if choices else ""  # extract correct answer string

#         arc_few_shots_list.append({
#             "query": data.get("query", ""),  # use 'query', not 'question'
#             "choices": choices,
#             "gold": gold_choice,
#             "answer_index": answer_index
#         })

printed_arc_prompt = False  # global flag

def arc_ai2_heb_prompt(line, task_name: str = "arc_ai2_heb"):
    global printed_arc_prompt

    # few_shot_text = ""
    # for i, shot in enumerate(arc_few_shots_list):
    #     qs = shot["query"].replace('\n', ' ').strip()
    #     choices = shot.get("choices", [])
    #     ans_index = shot.get("answer_index", 0)
    #     choices_text = ", ".join(choices)
    #     few_shot_text += (
    #         f"Question {i+1}: {qs}\n"
    #         f"Choices: {choices_text}\n"
    #         f"Right Index: {ans_index}\n\n"
    #     )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])
    target_choices_text = ", ".join(target_choices)
    instruction = f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
    full_prompt = f"""{instruction}
Question: {target_q}
Choices: {target_choices_text}
Answer:"""

    if not printed_arc_prompt:
        print("="*50)
        print("FULL ARC PROMPT WITH FEW-SHOTS (INTUITIVE FORMAT):")
        print(full_prompt)
        print("="*50)
        printed_arc_prompt = True

    return Doc(
        task_name=task_name,
        query=full_prompt,
        instruction=instruction,
        choices=target_choices,
        gold_index=line.get("answer_index", 0),
    )

# ============================================================
#  MMLU 
# ============================================================


def mmlu_heb_fewshot_prompt(line, task_name: str = "mmlu_heb"):
    """Few-shot prompt designed for loglikelihood evaluation"""


    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])
    topic = line.get("subject", "") # we dont use it now, becuase using it require fewshot per topic
    
    # Format target choices
    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    # For loglikelihood, we want a simple prompt that ends where the model should complete
    # the instruction in English:
    # query = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"

    # instruction = f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
    instruction = f"להלן שאלות רב-ברירה (עם תשובות)\n\n"
    prompt = (
        instruction +
        # f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    ).strip()

    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    # CRITICAL: For loglikelihood_acc, choices should be the completion options (A, B, C, D)
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]
    
    return Doc(
        task_name=task_name,
        query=prompt,
        # fewshot_samples = [
        #         Doc(query=shot["query"],
        #             choices=shot["choices"],
        #             gold_index=shot["answer_index"]) 
        #         for shot in mmlu_few_shots_list
        #     ],
        instruction=instruction,
        choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
        # choices=letter_choices,  # This should be ["A", "B", "C", "D"] not the actual answer texts
        gold_index=line["answer_index"],
    )

def mmlu(line, topic, task_name: str = None):
    query = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    )
# ============================================================
# COPA
# ============================================================

def xcopa(line, connectors: dict, task_name: str = None):
    connector = connectors[line["question"]]
    return Doc(
        task_name=task_name,
        query=f"{line['query'].strip()[:-1]} {connector}",
        choices=[f" {c}" for c in line["choices"]], 
        gold_index=int(line["answer_index"]),
    )


def xcopa_en(line, task_name: str = None):
    connectors = {"cause": "because", "effect": "therefore"}
    return xcopa(line, connectors, task_name)
def xcopa_heb(line, task_name: str = None):
    connectors = {"cause": "כי", "effect": "לכן"}
    return xcopa(line, connectors, task_name)

def full_copa_prompt(line, task_name: str = "copa"):
    global copa_prompt

    
    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    if not target_choices:
        print(f"COPA Warning: No choices found for target question: {target_q}")
        target_choices = ["A", "B"]  # fallback

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])
    instruction = f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A או B).\n"
    full_prompt = f"""{instruction}
        שאלה: {target_q}
{target_choices_formatted}
תשובה:"""


    # Get answer_index directly for validation data
    gold_index = line.get("answer_index")
    
    if gold_index is None:
        print(f"COPA Warning: No answer_index found in validation data for question: {target_q[:50]}")
        gold_index = 0
    elif not isinstance(gold_index, int):
        print(f"COPA Warning: answer_index must be integer, got {type(gold_index)}, using 0")
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices) or gold_index < 0:
        print(f"COPA Warning: Gold index {gold_index} out of range for {len(target_choices)} choices, using 0")
        gold_index = 0

    # Model predicts among A, B, C, D based on number of choices
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]
    
    # print(f"COPA DEBUG: Question='{target_q[:50]}...', Choices={len(target_choices)}, Gold_index={gold_index}, Expected={chr(65+gold_index)}")

    return Doc(
        task_name=task_name,
        query=full_prompt,
        instruction=instruction ,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# HellaSwag Hebrew 
# ============================================================

HELLASWAG_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/hellaswag_heb/train.jsonl"

hellaswag_few_shots_list = []

# print("Loading HellaSwag Hebrew few-shot examples...")
# with open(HELLASWAG_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
#     valid_examples = 0
#     for i, line in enumerate(f):
#         if valid_examples >= 5:  # take only 5 valid few-shots
#             break
        
#         try:
#             data = json.loads(line)
#             choices = data.get("choices", [])
            
#             if not choices:
#                 print(f"HellaSwag Warning: No choices found in line {i}, skipping")
#                 continue

#             # Get answer_index directly
#             answer_index = data.get("answer_index")
            
#             if answer_index is None:
#                 print(f"HellaSwag Warning: No answer_index found in line {i}, skipping")
#                 continue
            
#             if not isinstance(answer_index, int):
#                 print(f"HellaSwag Warning: answer_index must be integer, got {type(answer_index)} in line {i}, skipping")
#                 continue

#             # Validate the answer index is within range
#             if answer_index >= len(choices) or answer_index < 0:
#                 print(f"HellaSwag Warning: Answer index {answer_index} out of range for {len(choices)} choices in line {i}, skipping")
#                 continue

#             gold_choice = choices[answer_index]

#             hellaswag_few_shots_list.append({
#                 "query": data.get("query", ""),
#                 "choices": choices,
#                 "gold": gold_choice,
#                 "answer_index": answer_index,
#             })
            
#             valid_examples += 1
#             print(f"HellaSwag: Added few-shot example {valid_examples}: answer_index={answer_index} -> {chr(65+answer_index)}")
            
#         except json.JSONDecodeError as e:
#             print(f"HellaSwag Warning: JSON decode error in line {i}: {e}")
#             continue
#         except Exception as e:
#             print(f"HellaSwag Warning: Error processing line {i}: {e}")
#             continue

# print(f"HellaSwag: Successfully loaded {len(hellaswag_few_shots_list)} few-shot examples")

hellaswag_prompt = False

def hellaswag_heb_prompt(line, task_name: str = "hellaswag_heb"):
    global hellaswag_prompt

    # few_shot_text = ""
    # for i, shot in enumerate(hellaswag_few_shots_list):
    #     qs = shot["query"].replace('\n', ' ').strip()
    #     choices = shot.get("choices", [])
    #     ans_index = shot.get("answer_index", 0)
        
    #     # Format choices with letters (A, B for 2-choice)
    #     choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
    #     correct_letter = chr(65 + ans_index)

    #     few_shot_text += (
    #         f"שאלה {i+1}: {qs}\n"
    #         f"{choices_text}\n"
    #         f"תשובה: {correct_letter}\n\n"
    #     )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    # if not target_choices:
    #     print(f"HellaSwag Warning: No choices found for target question: {target_q}")
    #     target_choices = ["A", "B"]  # fallback

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    # Use A-D format for HellaSwag Hebrew (4 choices)
    instruction = f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
    full_prompt = f"""{instruction}
שאלה: {target_q}
{target_choices_formatted}
תשובה:"""

    if not hellaswag_prompt:
        print("="*50)
        print("FULL HELLASWAG HEBREW PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        hellaswag_prompt = True

    # Get answer_index directly for validation data
    gold_index = line.get("answer_index")
    
    if gold_index is None:
        print(f"HellaSwag Warning: No answer_index found in validation data for question: {target_q[:50]}")
        gold_index = 0
    elif not isinstance(gold_index, int):
        print(f"HellaSwag Warning: answer_index must be integer, got {type(gold_index)}, using 0")
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices) or gold_index < 0:
        print(f"HellaSwag Warning: Gold index {gold_index} out of range for {len(target_choices)} choices, using 0")
        gold_index = 0

    # Model predicts among A, B (or more if your data has more choices)
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]
    
    # print(f"HellaSwag DEBUG: Question='{target_q[:50]}...', Choices={len(target_choices)}, Gold_index={gold_index}, Expected={chr(65+gold_index)}")

    return Doc(
        task_name=task_name,
        query=full_prompt,
        instruction=instruction,
        choices=letter_choices,
        gold_index=gold_index,
    )

# ============================================================
# Psychometric Math Section
# ============================================================

PSYCHOMETRIC_MATH_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/math/train.jsonl"

psychometric_math_few_shots_list = []

with open(PSYCHOMETRIC_MATH_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_math_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_math_prompt = False

def psychometric_test_math_prompt(line, task_name: str = "psychometric_test_math"):
    global printed_psycho_math_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_math_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_math_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC MATH PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_math_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )


# ============================================================
# Psychometric Analogies Section
# ============================================================

PSYCHOMETRIC_ANALOGIES_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/analogies/train.jsonl"

psychometric_analogies_few_shots_list = []

with open(PSYCHOMETRIC_ANALOGIES_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_analogies_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_analogies_prompt = False

def psychometric_test_analogies_prompt(line, task_name: str = "psychometric_test_analogies"):
    global printed_psycho_analogies_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_analogies_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_analogies_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC ANALOGIES PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_analogies_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Restatement Section
# ============================================================

PSYCHOMETRIC_RESTATEMENT_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/restatement/train.jsonl"

psychometric_restatement_few_shots_list = []

with open(PSYCHOMETRIC_RESTATEMENT_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_restatement_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_restatement_prompt = False

def psychometric_test_restatement_prompt(line, task_name: str = "psychometric_test_restatement"):
    global printed_psycho_restatement_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_restatement_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_restatement_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC RESTATEMENT PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_restatement_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Sentence Complete English Section
# ============================================================

PSYCHOMETRIC_SENTENCE_COMPLETE_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/sentence_complete_english/train.jsonl"

psychometric_sentence_complete_few_shots_list = []

with open(PSYCHOMETRIC_SENTENCE_COMPLETE_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_sentence_complete_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_sentence_complete_prompt = False

def psychometric_test_sentence_complete_english_prompt(line, task_name: str = "psychometric_test_sentence_complete"):
    global printed_psycho_sentence_complete_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_sentence_complete_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_sentence_complete_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC SENTENCE COMPLETE ENGLISH PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_sentence_complete_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Sentence Complete Hebrew Section
# ============================================================

PSYCHOMETRIC_SENTENCE_COMPLETE_HEBREW_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/sentence_complete_hebrew/train.jsonl"

psychometric_sentence_complete_hebrew_few_shots_list = []

with open(PSYCHOMETRIC_SENTENCE_COMPLETE_HEBREW_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_sentence_complete_hebrew_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_sentence_complete_hebrew_prompt = False

def psychometric_test_sentence_complete_hebrew_prompt(line, task_name: str = "psychometric_test_sentence_complete_hebrew"):
    global printed_psycho_sentence_complete_hebrew_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_sentence_complete_hebrew_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_sentence_complete_hebrew_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC SENTENCE COMPLETE HEBREW PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_sentence_complete_hebrew_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Text English Section
# ============================================================

PSYCHOMETRIC_TEXT_ENGLISH_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/text_english/train.jsonl"

psychometric_text_english_few_shots_list = []

with open(PSYCHOMETRIC_TEXT_ENGLISH_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_text_english_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_text_english_prompt = False

def psychometric_test_text_english_prompt(line, task_name: str = "psychometric_test_text_english"):
    global printed_psycho_text_english_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_text_english_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_text_english_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC TEXT ENGLISH PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_text_english_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Text Hebrew Section
# ============================================================

PSYCHOMETRIC_TEXT_HEBREW_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/text_hebrew/train.jsonl"

psychometric_text_hebrew_few_shots_list = []

with open(PSYCHOMETRIC_TEXT_HEBREW_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_text_hebrew_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_text_hebrew_prompt = False

def psychometric_test_text_hebrew_prompt(line, task_name: str = "psychometric_test_text_hebrew"):
    global printed_psycho_text_hebrew_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_text_hebrew_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_text_hebrew_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC TEXT HEBREW PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_text_hebrew_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Psychometric Understanding Hebrew Section
# ============================================================

PSYCHOMETRIC_UNDERSTANDING_HEBREW_FEW_SHOT_PATH = f"{HEB_BENCHMARKS_DIR_PATH}/hebrew_benchmarks_data_final/psychometric_heb/understanding_hebrew/train.jsonl"

psychometric_understanding_hebrew_few_shots_list = []

with open(PSYCHOMETRIC_UNDERSTANDING_HEBREW_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # take only 5 few-shots
            break
        data = json.loads(line)
        choices = data.get("choices", [])

        # Better answer mapping with validation
        answer_raw = data.get("answer", "A")
        if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer_raw.upper()) - ord("A")
            # Validate the answer index is within range
            if answer_index >= len(choices):
                answer_index = 0  # fallback
        elif isinstance(answer_raw, int):
            answer_index = answer_raw
        else:
            answer_index = 0

        gold_choice = choices[answer_index] if 0 <= answer_index < len(choices) else ""

        psychometric_understanding_hebrew_few_shots_list.append({
            "query": data.get("query", ""),
            "choices": choices,
            "gold": gold_choice,
            "answer_index": answer_index,
            "raw_answer": answer_raw
        })

printed_psycho_understanding_hebrew_prompt = False

def psychometric_test_understanding_hebrew_prompt(line, task_name: str = "psychometric_test_understanding_hebrew"):
    global printed_psycho_understanding_hebrew_prompt

    few_shot_text = ""
    for i, shot in enumerate(psychometric_understanding_hebrew_few_shots_list):
        qs = shot["query"].replace('\n', ' ').strip()
        choices = shot.get("choices", [])
        ans_index = shot.get("answer_index", 0)
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        correct_letter = chr(65 + ans_index)

        few_shot_text += (
            f"שאלה {i+1}: {qs}\n"
            f"{choices_text}\n"
            f"תשובה: {correct_letter}\n\n"
        )

    target_q = line["query"].replace('\n', ' ').strip()
    target_choices = line.get("choices", [])

    target_choices_formatted = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(target_choices)])

    full_prompt = (
        f"ענה על השאלות הבאות על ידי בחירת האות המתאימה (A, B, C, או D).\n\n"
        f"{few_shot_text}"
        f"שאלה: {target_q}\n"
        f"{target_choices_formatted}\n"
        f"תשובה:"
    )

    if not printed_psycho_understanding_hebrew_prompt:
        print("="*50)
        print("FULL PSYCHOMETRIC UNDERSTANDING HEBREW PROMPT WITH FEW-SHOTS:")
        print(full_prompt)
        print("="*50)
        printed_psycho_understanding_hebrew_prompt = True

    # Better answer processing for validation data
    answer_raw = line.get("answer", "A")
    if isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gold_index = ord(answer_raw.upper()) - ord("A")
    elif isinstance(answer_raw, int):
        gold_index = answer_raw
    else:
        gold_index = 0

    # Validate gold index is within range
    if gold_index >= len(target_choices):
        gold_index = 0

    # Model predicts among A, B, C, D
    letter_choices = [chr(65 + i) for i in range(len(target_choices))]

    return Doc(
        task_name=task_name,
        query=target_q,
        instruction=full_prompt,
        choices=letter_choices,
        gold_index=gold_index,
    )
# ============================================================
# Task configs
# ============================================================

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

_TASKS = [
    # LightevalTaskConfig(
    #     name="gsm8k_heb",
    #     suite=["community"],
    #     prompt_function=gsm8k_heb_fewshot_prompt,  # <-- updated to few-shot version
    #     hf_subset="default",
    #     metric=[GSM8KFinalAccMetric],
    #     hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "gsm8k_heb/"),
    #     evaluation_splits=["validation"],
    #     stop_sequence=['[ANSWER_END]'],
    #     generation_size=512
    # ),
    LightevalTaskConfig(
        name="arc_ai2_heb",
        suite=["community"],
        prompt_function=arc_ai2_heb_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "arc_ai2_heb/"),
        evaluation_splits=["validation"],
        few_shots_split="train",
        generation_size=32
    ),
LightevalTaskConfig(
    name="mmlu_heb",
    suite=["community"],
    prompt_function=mmlu_heb_fewshot_prompt,
    hf_subset="default",
    metrics=[Metrics.loglikelihood_acc],  # Use loglikelihood_acc
    hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "mmlu_heb/"),
    evaluation_splits=["validation"],
    few_shots_split="train",
    must_remove_duplicate_docs=True,
    # No generation_size or stop_sequence needed for loglikelihood
),
    LightevalTaskConfig(
        name="psychometric_heb_math",
        suite=["community"],
        prompt_function=psychometric_test_math_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "math"),
        evaluation_splits=["validation"],
    ),
       LightevalTaskConfig(
        name="psychometric_heb_analogies",
        suite=["community"],
        prompt_function=psychometric_test_analogies_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "analogies"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
       LightevalTaskConfig(
        name="psychometric_heb_restatement",
        suite=["community"],
        prompt_function=psychometric_test_restatement_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "restatement"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
       LightevalTaskConfig(
        name="psychometric_heb_sentence_complete_english",
        suite=["community"],
        prompt_function=psychometric_test_sentence_complete_english_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "sentence_complete_english"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
LightevalTaskConfig(
        name="psychometric_heb_sentence_complete_hebrew",
        suite=["community"],
        prompt_function=psychometric_test_sentence_complete_hebrew_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "sentence_complete_hebrew"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
    LightevalTaskConfig(
        name="psychometric_heb_sentence_text_english",
        suite=["community"],
        prompt_function=psychometric_test_text_english_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "text_english"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
     LightevalTaskConfig(
        name="psychometric_heb_sentence_text_hebrew",
        suite=["community"],
        prompt_function=psychometric_test_text_hebrew_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "text_hebrew"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
    LightevalTaskConfig(
        name="psychometric_heb_understanding_hebrew",
        suite=["community"],
        prompt_function=psychometric_test_understanding_hebrew_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "psychometric_heb", "understanding_hebrew"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),
       LightevalTaskConfig(
        name="copa_heb",
        suite=["community"],
        prompt_function= full_copa_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "copa_heb/"),
        evaluation_splits=["validation"],
        few_shots_split="train",
        # generation_size=-1,
        # metric=[
        #     loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        # ],
    ),
     LightevalTaskConfig(
        name="hellaswag_heb",
        suite=["community"],
        prompt_function= hellaswag_heb_prompt,
        hf_subset="default",
        metrics=[Metrics.loglikelihood_acc],  # multiple choice accuracy
        hf_repo=os.path.join(_CURR_DIR, "..", "hebrew_benchmarks_data_final", "hellaswag_heb/"),
        evaluation_splits=["validation"],
        few_shots_split="train",
    ),

]
TASKS_TABLE = _TASKS
__all__ = ["TASKS_TABLE"]