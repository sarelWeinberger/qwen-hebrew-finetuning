import regex

html_tags = r'</?(html|head|body|style|script|title|meta|link|div|span|p|a|img|br|h[1-6]|ul|ol|li|table|thead|tbody|tr|th|td|form|input|button|label|select|option|textarea|strong|b|em|i|u|small|footer|header|nav|section|article|aside|main|figure|figcaption|hr)[^>]*>'
CLEANUP_RULES = [
    {
        'regex': (r'(&quot;|&#34;)', '"'),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/html_escape_codes/',
        'info': 'html_escape_codes - Replaces HTML escape codes such as &quot;, &#34;, and &#39; with their respective characters.'
    },
    {
        'regex': (r'&#39;', "'"),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/html_escape_codes/',
        'info': 'html_escape_codes - Replaces HTML escape codes such as &quot;, &#34;, and &#39; with their respective characters.'
    },
    {
        'regex': (r'<style[^>]*>[^<]*</style>', ''),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/CSS/',
        'info': 'Remove CSS (between <style> tags) and HTML tags'
    },
    {
        'regex': (html_tags, ''),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/CSS/',
        'info': 'Remove CSS (between <style> tags) and HTML tags'
    },
    {
        'regex': (r'^\s*$', '', regex.MULTILINE),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {
        'regex': (r'\r\n?', '\n'),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {
        'regex': (r'\n{4,}', '\n\n\n'),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/cr_and_3_newlines/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {    
        'regex': (r'^(( {4})+(?!\s))|^\s+|\s+$|( ) +', r'\1\3', regex.MULTILINE),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/multiple_spaces/',
        'info': 'Multiple space: replace multiple space to its lower 4 multiplier (10->8, 12->12, 7->4), until a max of 16 (28->16). x//4 . exclude for markdown tables.'
    },
    {
        'regex': (r'\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>'),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/PII/',
        'info': 'PII deletion – IP delete except local list, delete mail addresses.'
    },
    {
        'regex': (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '<email>'),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/PII/',
        'info': 'PII deletion – IP delete except local list, delete mail addresses.'
    },
    {
       'regex': (r'^[\-=*_~•●■▪◆◦\s-[\n]]+$', '', regex.MULTILINE),
       'bucket_name': 'gepeta-datasets',
       'path': 'partly-processed/round_2_test_examples/multiple_hyphens/',
       'info': 'Remove long separator lines made of multiple hyphens or similar symbols.'
    },
    {
        'regex': (r'[\u0591-\u05C7-[\u05be]]', ''),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/nikud/',
        'info': '05be it upper dash which we want to keep'
    },
    {
        'regex': (r'\u00A0', ' '),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/nikud/',
        'info': ''
    },
    {
        'regex': (r'[\u200E\u200F\u202A-\u202F]', ''),
        'bucket_name': 'gepeta-datasets',
        'path': 'partly-processed/round_2_test_examples/nikud/',
        'info': ''
    }

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


