# 1. inference.py - קוד ה-inference שיכנס ל-container
"""
קובץ: inference.py
מטרה: הקוד שיעבוד בתוך ה-container של SageMaker
"""

import json
import torch
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import login


class TextCleaningModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False

    def load_model(self):
        """טעינת המודל - יקרא פעם אחת בהתחלת הcontainer"""
        if self.loaded:
            return

        print("🔄 Loading model...")

        # קריאת HF token מ-environment variable
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(hf_token)

        model_name = "google/gemma-3-27b-it"

        # הגדרת quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # טעינת המודל
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        ).eval()

        # טעינת הprocessor
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.loaded = True
        print("✅ Model loaded successfully")

    def clean_text(self, text, max_new_tokens=200):
        """ניקוי טקסט בודד"""
        if not self.loaded:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text",
                             "text": "אתה עוזר לניקוי טקסטים עבריים. קבל טקסטים עבריים רועשים שעשויים להכיל פגמי קידוד (&quot;), קטעי HTML, טלפון/אימייל, אימוג'ים, פרסומות, או תבניות. החזר רק את הטקסט המנוקה, תוך שמירה על המשמעות, בעברית, ללא הסברים."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""נקה טקסטים עבריים מפגמי קידוד, תבניות HTML, פרסומות, מידע מיותר ותגיות.

דוגמה 1:
קלט: דיווח: תא דאעש שנחשף בירדן תכנן לפגוע באנשי עסקים ישראליים
© סופק על ידי מעריב תא דאעש... ____________________________________________________________ סרטונים שווים ב-MSN (BuzzVideos)
פלט: דיווח: תא דאעש שנחשף בירדן תכנן לפגוע באנשי עסקים ישראליים
תא דאעש שנחשף בנובמבר האחרון בירדן, תכנן בין היתר לפגוע באנשי עסקים ישראלים ברבת עמון...

דוגמה 2:
קלט: סוחר שהפיץ נפצים באשדוד וערים אחרות הופלל בוואטסאפ
אלה רוזנבלט... היי, בלוח החדש של אשדוד נט כבר ביקרת? כל הדירות למכירה/השכרה באשדוד... אולי יעניין אותך גם
פלט: סוחר שהפיץ נפצים באשדוד וערים אחרות הופלל בוואטסאפ
אלה רוזנבלט
מחירון לנפצים שהופץ באפליקציה ע"י צעיר ירושלמי הביא לתפיסתו בעת ביצוע העסקה...

עכשיו נקה את הטקסט הבא:
{text}

השב רק עם הטקסט המנוקה:"""}
                ]
            }
        ]

        try:
            # עיבוד ההודעות
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            # העברה למכשיר הנכון
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[-1]

            # יצירת תגובה
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )

            # פענוח רק החלק החדש
            generated_ids = outputs[0][input_len:]
            decoded = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return decoded.strip()

        except Exception as e:
            print(f"Error in generation: {e}")
            return f"[ERROR] {str(e)}"


# יצירת instance גלובלי
model_instance = TextCleaningModel()


def model_fn(model_dir):
    """SageMaker model loading function"""
    model_instance.load_model()
    return model_instance


def input_fn(request_body, request_content_type):
    """SageMaker input processing function"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(data, model):
    """SageMaker prediction function"""
    if isinstance(data, dict):
        # batch של טקסטים
        if 'texts' in data:
            texts = data['texts']
            results = []
            for i, text_item in enumerate(texts):
                text = text_item.get('text', '') if isinstance(text_item, dict) else text_item
                print(f"Processing text {i + 1}/{len(texts)}")
                cleaned = model.clean_text(text)
                results.append({
                    'index': i,
                    'original': text,
                    'cleaned': cleaned
                })
            return {'results': results}

        # טקסט בודד
        elif 'text' in data:
            text = data['text']
            cleaned = model.clean_text(text)
            return {'cleaned': cleaned}

    raise ValueError("Invalid input format. Expected 'text' or 'texts' key.")


def output_fn(prediction, accept):
    """SageMaker output processing function"""
    if accept == 'application/json':
        return json.dumps(prediction, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# לבדיקה מקומית
if __name__ == "__main__":
    # טסט מקומי
    test_data = {
        "texts": [
            {"text": "דרעי: אין סיבה שניכנס לעימותים בקואליציה סביב חוק הגיוס. תגיות: דרעי חוק גיוס"},
            {"text": "חדשות ספורט: הפועל ניצחה!!! © כל הזכויות שמורות... Follow @sport"}
        ]
    }

    model = model_fn("/opt/ml/model")
    prediction = predict_fn(test_data, model)
    result = output_fn(prediction, 'application/json')
    print(result)