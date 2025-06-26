import boto3
import json
from google import genai
from google.genai import types

CLAUDE_MODEL = "arn:aws:bedrock:us-east-1:670967753077:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
GEMINI_MODEL = "gemini-2.5-flash-lite-preview-06-17"


def bedrock_connect(acceess_id, secret_key):
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',  # Change to your region
        aws_access_key_id=acceess_id,
        aws_secret_access_key=secret_key,
    )


# Claude call
def call_claude_bedrock(bedrock_client, message, max_tokens=300):
    """
    Call Claude through AWS Bedrock
    """
    # Prepare the request body
    body = json.dumps({
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
    })

    # Make the API call
    response = bedrock_client.invoke_model(
        body=body,
        # modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        # modelId='anthropic.claude-3-7-sonnet-20250219-v1:0',
        modelId=CLAUDE_MODEL,
        contentType='application/json',
        accept='application/json'
    )

    # Parse and return response
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']


# Gemini call
def google_connect(key):
    return genai.Client(api_key=key)


def call_gemini(google_client, message, config=None):
    response = google_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=message,
        config=config,
    )
    return response


def all_string_gemini_config(str_lst, system_instruction=None):
    return types.GenerateContentConfig(
        temperature=0,
        topK=1,
        thinking_config=types.ThinkingConfig(
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
                    type=genai.types.Type.STRING,
                ) for p in str_lst
            },
        ),
    )    