import requests
import json
from time import sleep
import gradio as gr

"""
This code is (mostly) brought to you by Gemini.
"""


def get_wikidata_qid(term_en):
    """
    Finds the Wikidata QID (item ID) for a given English term.
    
    Args:
        term_en (str): The English scientific term to search for.
        
    Returns:
        str: The Wikidata QID (e.g., "Q7430") or None if not found.
    """
    # The Wikidata API endpoint for searches
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": term_en,
        "language": "en",
        "type": "item",
        "limit": 1
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an error for bad status codes
        data = response.json()
        
        if data["search"]:
            # Return the QID of the first result
            return data["search"][0]["id"]
        else:
            print(f"No Wikidata item found for '{term_en}'.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error searching for term '{term_en}': {e}")
        return None

def translate_term(term_en, target_langs=["he"]):
    """
    Translates a scientific term from English to a list of other languages
    using the Wikidata SPARQL API, retrieving both the label and description.

    Args:
        term_en (str): The English scientific term (e.g., "DNA").
        target_langs (list): A list of language codes (e.g., ["es", "fr", "de"]).

    Returns:
        dict: A dictionary where each key is a target language, and the value is
              another dictionary with "label" and "description".
              Returns an empty dictionary if the term is not found or an error occurs.
    """
    print(f"Attempting to find Wikidata item for: '{term_en}'...")
    qid = get_wikidata_qid(term_en)
    
    if not qid:
        return {}

    print(f"Found QID: {qid}. Fetching translations and descriptions...")

    # The SPARQL query endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    
    # Construct the SPARQL query string
    # We now select ?label and ?description
    query = f"""
    SELECT ?lang ?label ?description
    WHERE 
    {{
      BIND(wd:{qid} AS ?item)
      ?item rdfs:label ?label.
      ?item schema:description ?description.
      
      BIND(LANG(?label) AS ?lang_label)
      BIND(LANG(?description) AS ?lang_desc)
      
      FILTER(?lang_label IN ("{'", "'.join(target_langs)}"))
      FILTER(?lang_desc = ?lang_label)
      BIND(?lang_label AS ?lang)
    }}
    """
    
    # Set the headers to request a JSON response
    headers = {
        'Accept': 'application/sparql-results+json'
    }

    try:
        # Exponential backoff to handle potential throttling
        max_retries = 5
        retry_delay = 1  # seconds
        for i in range(max_retries):
            response = requests.get(endpoint_url, headers=headers, params={'query': query})
            if response.status_code == 200:
                break
            print(f"Request failed with status code {response.status_code}. Retrying in {retry_delay}s...")
            sleep(retry_delay)
            retry_delay *= 2
        else:
            response.raise_for_status()

        data = response.json()
        
        translations = {}
        for result in data["results"]["bindings"]:
            lang_code = result["lang"]["value"]
            label = result["label"]["value"]
            description = result["description"]["value"]
            
            if lang_code not in translations:
                translations[lang_code] = {}
            
            translations[lang_code]['label'] = label
            translations[lang_code]['description'] = description
        
        return translations
    
    except requests.exceptions.RequestException as e:
        print(f"Error translating term '{term_en}': {e}")
        return {}

# Example usage:
if __name__ == '__main__':
    english_term = "DNA"
    hebrew_translation = translate_term(english_term, target_langs=["he"])
    
    if hebrew_translation and "he" in hebrew_translation:
        print(f"\n--- Translation for '{english_term}' ---")
        print(f"Language: he")
        print(f"  Label: {hebrew_translation['he'].get('label', 'N/A')}")
        print(f"  Description: {hebrew_translation['he'].get('description', 'N/A')}")
    else:
        print(f"\nCould not retrieve the Hebrew translation and description for '{english_term}'.")

    print("\n" + "="*30 + "\n")

    english_term_astro = "Black hole"
    multi_language_translations = translate_term(english_term_astro, target_langs=["es", "fr", "de"])

    if multi_language_translations:
        print(f"--- Translations for '{english_term_astro}' ---")
        for lang, data in multi_language_translations.items():
            print(f"Language: {lang}")
            print(f"  Label: {data.get('label', 'N/A')}")
            print(f"  Description: {data.get('description', 'N/A')}")
    else:
        print(f"\nCould not retrieve translations for '{english_term_astro}'.")


def translate_try_multi(x):
    lst = []
    for v in translate_term(x).values():
        lst.append(v['label'] + ' - ' + v['description'])
    for v in translate_term(x + ' (math').values():
        lst.append(v['label'] + ' - ' + v['description'])
    for v in translate_term(x + ' (mathematics)').values():
        lst.append(v['label'] + ' - ' + v['description'])
        
    lst += ['']

    return '\n'.join(lst)


def wikidata_gradio():
    # --- Create the Gradio Interface ---
    iface = gr.Interface(
     fn=translate_try_multi,
     inputs=gr.Textbox(lines=2, placeholder="Enter term to translate..."),
     outputs="text",
     title="Term Translator",
     description="Enter a term and click 'Run' to see the translation."
    )

    iface.launch(share=True, inline=False)
    return iface
