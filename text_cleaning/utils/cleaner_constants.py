# Define the regex patterns for cleaning
CLEANUP_RULES = [
    # ——— HTML / placeholders ———
    (r'<[^>]+>', ''),                     # HTML tags or placeholders like <email>, <phone>, <url> ...
    
    # ——— Links and references ———
    (r'http[s]?://\S+', ''),              # URLs
    (r'www\.\S+',     ''),                # URLs without http
    (r'Follow\s+@[\w_]+', ''),            # "Follow @user" instructions
    (r'(שתף|Share).*$' , ''),             # Share buttons and social widgets
    
    # ——— Footnotes / references ———
    (r'\[[0-9]+\]',  ''),                 # [1], [23] ...
    (r'\([0-9]+\)',  ''),                 # (1) — sometimes appears this way
    
    # ——— Credit lines / rights / advertisements ———
    (r'©.*?(?:\n|$)', ''),                # Lines starting with ©
    (r'.*?(לחץ כאן|Click here).*$' , ''), # "Click here" advertisements
    (r'>>.*?>>',      ''),                # Advertisements within >> ... >>
    
    # ——— Bullet points, separators and other graphic symbols ———
    (r'[\u2022\u25CF\u25AA\u25E6]', ''),  # • ● ■ ◦
    (r'[_\-–—]{2,}',  ''),                # Long separator lines ___ --- ——
    (r'[■▪◆]',        ''),                # Additional bullet points
    (r'\.{3,}',       '…'),               # Multiple dots → single ellipsis
    
    # ——— Space and quote cleanup ———
    (r'""',           '"'),               # "" → "
    (r'\s{2,}',       ' '),               # Multiple spaces → single space
    (r'\n{3,}',       '\n\n'),            # Multiple empty lines → two lines
]

# Define the source names
SOURCES = [
    "AllHebNLIFiles-Deduped-D2.forgpt",
    "AllOfHEOscarData-Combined-Deduped-DC4.forgpt",
    "AllOfNewHebrewWikipediaWithArticles-Oct29-2023.forgpt",
    "AllTzenzuraData-Combined-Deduped-DC4.forgpt",
    "BooksNLI2-Combined-Deduped.forgpt",
    "GeektimeCorpus-Combined-Deduped.forgpt",
    "hebrew_tweets_text_clean_full-Deduped.forgpt",
    "HeC4DictaCombined-Clean-Deduped.forgpt",
    "YifatDataBatch2-Round3-Deduped.forgpt",
    "YifatDataRound2-Deduped.forgpt",
    "YifatToCombine-Deduped.forgpt",
    "YisraelHayomData-Combined-Deduped.forgpt"
] 