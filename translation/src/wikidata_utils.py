import requests
import json
from time import sleep

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
            print()
            print(len(data["search"]))
            print()
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
    using the Wikidata SPARQL API.

    Args:
        term_en (str): The English scientific term (e.g., "DNA").
        target_langs (list): A list of language codes (e.g., ["es", "fr", "de"]).

    Returns:
        dict: A dictionary with the target language as the key and the
              translation as the value. Returns an empty dictionary if
              the term is not found or an error occurs.
    """
    print(f"Attempting to find Wikidata item for: '{term_en}'...")
    qid = get_wikidata_qid(term_en)
    
    if not qid:
        return {}

    print(f"Found QID: {qid}. Fetching translations...")

    # The SPARQL query endpoint
    endpoint_url = "https://query.wikidata.org/sparql"
    
    # Construct the SPARQL query string
    # We use a FILTER to only get labels in the languages we care about.
    # The SERVICE statement allows us to get the labels directly.
    query = f"""
    SELECT ?lang ?label
    WHERE 
    {{
      BIND(wd:{qid} AS ?item)
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],{','.join(target_langs)}" .
        ?item rdfs:label ?label .
      }}
      BIND(LANG(?label) AS ?lang)
      FILTER(?lang IN ("{'", "'.join(target_langs)}"))
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
            translations[lang_code] = label
        
        return translations
    
    except requests.exceptions.RequestException as e:
        print(f"Error translating term '{term_en}': {e}")
        return {}


def check_wikidata_transalte():
    # Example usage
    english_term = "Mitochondrion"
    target_languages = ["he"]

    translations = translate_term(english_term, target_languages)

    if translations:
        print(f"\nTranslations for '{english_term}':")
        for lang, translation in translations.items():
            print(f"- {lang.upper()}: {translation}")
    else:
        print(f"\nCould not get translations for '{english_term}'.")

    print("-" * 30)

    english_term_2 = "Quantum entanglement"
    target_languages_2 = ["he"]
    translations_2 = translate_term(english_term_2, target_languages_2)

    if translations_2:
        print(f"\nTranslations for '{english_term_2}':")
        for lang, translation in translations_2.items():
            print(f"- {lang.upper()}: {translation}")
    else:
        print(f"\nCould not get translations for '{english_term_2}'.")
