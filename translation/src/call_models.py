import boto3
import json
from google import genai
from google.genai import types

CLAUDE_MODEL_3 = "arn:aws:bedrock:us-east-1:670967753077:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# CLAUDE_MODEL_4 = "arn:aws:bedrock:us-east-1:670967753077:inference-profile/us.anthropic.claude-opus-4-20250514-v1:0"
CLAUDE_MODEL_4 = "arn:aws:bedrock:us-east-1:670967753077:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
GEMINI_MODEL_PRO = "gemini-2.5-pro"
GEMINI_MODEL_FLASH = "gemini-2.5-flash"

GEMMA_MODEL = "gemma-3-27b-it"


def bedrock_connect(acceess_id, secret_key):
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',  # Change to your region
        aws_access_key_id=acceess_id,
        aws_secret_access_key=secret_key,
    )


def call_claude_bedrock_with_thinking(bedrock_client, message, system_prompt=None, max_tokens=12_000, thinking_budget=2000, if_four=False):
    """
    Call Claude through AWS Bedrock with extended thinking enabled.
    """
    if if_four:
        model_name = CLAUDE_MODEL_4
    else:
        model_name = CLAUDE_MODEL_3
    # Prepare the request body
    claude_params = {
        "anthropic_version": "bedrock-2023-05-31",
        # "top_k": 1,
        # "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": message,
            }]
        }],
        "thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
    }
    if system_prompt:
        claude_params['system'] = system_prompt
    body = json.dumps(claude_params)

    # Make the API call
    response = bedrock_client.invoke_model(
        body=body,
        modelId=model_name,
        contentType='application/json',
        accept='application/json'
    )

    # Parse and return response
    response_body = json.loads(response.get('body').read())
    
    # The response will now contain both "thinking" and "content" blocks.
    # You can process the thinking steps and the final answer separately.
    thinking_content = [item for item in response_body.get('content', []) if item.get('type') == 'thinking']
    final_content = [item for item in response_body.get('content', []) if item.get('type') == 'text']

    # For this example, we'll return the thinking process and the final text.
    # In a real application, you might want to handle these differently.
    output = final_content[0]['text'] if final_content else ""
    # Return (text, thinking)
    return output,  thinking_content[0]['thinking']

# Claude call
def call_claude_bedrock(bedrock_client, message, system_prompt=None, max_tokens=10_000):
    """
    Call Claude through AWS Bedrock
    """
    # Prepare the request body
    claude_params = {
        "anthropic_version": "bedrock-2023-05-31",
        "top_k": 1,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": message,
                "cache_control": {"type": "ephemeral"}
            }]
        }]
    }
    if system_prompt:
        claude_params['system'] = system_prompt
    body = json.dumps(claude_params)

    # Make the API call
    response = bedrock_client.invoke_model(
        body=body,
        modelId=CLAUDE_MODEL,
        contentType='application/json',
        accept='application/json'
    )

    # Parse and return response
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']


# Gemini call
def google_connect(key, project_id=None):
    return genai.Client(api_key=key, project=project_id)


def call_gemini(google_client, message, config=None, if_pro=False):
    if if_pro:
        model_name = GEMINI_MODEL_PRO
    else:
        model_name = GEMINI_MODEL_FLASH
    response = google_client.models.generate_content(
        model=model_name,
        contents=message,
        config=config,
    )
    return response


def call_gemma(google_client, message):
    response = google_client.models.generate_content(
        model=GEMMA_MODEL,
        contents=message,
        config=types.GenerateContentConfig(
            temperature=1,
            # topK=1,
        ),
    )
    return response


def create_gemini_config(str_lst, system_instruction=None, output_type=None, enum=None, think_bud=-1, temp=0):
    if output_type is None:
        output_type = types.Type.STRING

    return types.GenerateContentConfig(
        temperature=temp,
        # topK=1,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=think_bud,
        ),
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=str_lst,
            propertyOrdering=str_lst,
            properties={
                p: genai.types.Schema(
                    type=output_type,
                    enum=enum,
                ) for p in str_lst
            },
        ),
    )


def all_string_gemini_config(str_lst, system_instruction=None, enum=None, think_bud=-1, temp=0):
    return create_gemini_config(
        str_lst,
        system_instruction=system_instruction,
        enum=enum,
        think_bud=think_bud,
        temp=temp,
    )


def all_int_gemini_config(str_lst, system_instruction=None, think_bud=-1):
    return create_gemini_config(
        str_lst,
        system_instruction=system_instruction,
        output_type=types.Type.INTEGER,
        think_bud=think_bud,
    )


def all_list_gemini_config(str_lst, system_instruction=None, length=3):
    return types.GenerateContentConfig(
        temperature=0,
        topK=1,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        ),
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=str_lst,
            propertyOrdering=str_lst,
            properties={
                p: genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING
                    ),
                    minItems=length,
                    maxItems=length,
                ) for p in str_lst
            },
        ),
    )