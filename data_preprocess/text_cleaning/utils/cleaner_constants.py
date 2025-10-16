import regex

# Constants
CLEAN_PATH = "round_2_test_examples"

html_tags = r'</?(html|head|body|style|script|title|meta|link|div|span|p|a|img|br|h[1-6]|ul|ol|li|table|thead|tbody|tr|th|td|form|input|button|label|select|option|textarea|strong|b|em|i|u|small|footer|header|nav|section|article|aside|main|figure|figcaption|hr)[^>]*>'

# Helper function to create regex that excludes MARKDOWN_TABLE content
def exclude_markdown_table(regex_pattern):
    """
    Wraps a regex pattern to exclude content within MARKDOWN_TABLE tags.
    Uses negative lookbehind and negative lookahead to ensure the match
    is not within <MARKDOWN_TABLE>...</MARKDOWN_TABLE> tags.
    """
    return f'(?<!<MARKDOWN_TABLE[^>]*>)(?<!<MARKDOWN_TABLE>){regex_pattern}(?!</MARKDOWN_TABLE>)'

CLEANUP_RULES = [
    {
        'regex': (exclude_markdown_table(r'(&quot;|&#34;)'), '"'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/html_escape_codes/',
        'info': 'html_escape_codes - Replaces HTML escape codes such as &quot;, &#34;, and &#39; with their respective characters.'
    },
    {
        'regex': (exclude_markdown_table(r'&#39;'), "'"),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/html_escape_codes/',
        'info': 'html_escape_codes - Replaces HTML escape codes such as &quot;, &#34;, and &#39; with their respective characters.'
    },
    {
        'regex': (exclude_markdown_table(r'<style[^>]*>[^<]*</style>'), ''),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/CSS/',
        'info': 'Remove CSS (between <style> tags) and HTML tags'
    },
    {
        'regex': (exclude_markdown_table(html_tags), ''),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/CSS/',
        'info': 'Remove CSS (between <style> tags) and HTML tags'
    },
    {
        'regex': (exclude_markdown_table(r'^\s*$'), '', regex.MULTILINE),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {
        'regex': (exclude_markdown_table(r'\r\n?'), '\n'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {
        'regex': (exclude_markdown_table(r'\n{4,}'), '\n\n\n'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/cr_and_3_newlines/',
        'info': 'Newline and spaces handling: Remove carriage return (\r), and replace more than 3 consecutive new lines to maximum of 3'
    },
    {    
        'regex': (exclude_markdown_table(r'^(( {4})+(?!\s))|^\s+|\s+$|( ) +'), r'\1\3', regex.MULTILINE),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/multiple_spaces/',
        'info': 'Multiple space: replace multiple space to its lower 4 multiplier (10->8, 12->12, 7->4), until a max of 16 (28->16). x//4 . exclude for markdown tables.'
    },
    {
        'regex': (exclude_markdown_table(r'\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}\b'), '<IP>'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/PII/',
        'info': 'PII deletion – IP delete except local list, delete mail addresses.'
    },
    {
        'regex': (exclude_markdown_table(r'\b[A-Za-z0-9!#$%&\'+/=?^`{|}-]+(?:\.[A-Za-z0-9!#$%&\'+/=?^`{|}-]+)*@(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-][A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-][A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)\])\b'), '<EMAIL>'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/PII/',
        'info': 'PII deletion – Comprehensive email pattern matching including special characters and IP-based email addresses.'
    },
    {
        'regex': (exclude_markdown_table(r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'), '<IP>'),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/PII/',
        'info': 'PII deletion – Comprehensive IP address pattern matching for all valid IP formats.'
    },
    {
       'regex': (exclude_markdown_table(r'^[\-=*_~•●■▪◆◦\s-[\n]]+$'), '', regex.MULTILINE),
       'bucket_name': 'gepeta-datasets',
       'path': f'partly-processed/{CLEAN_PATH}/multiple_hyphens/',
       'info': 'Remove long separator lines made of multiple hyphens or similar symbols.'
    },
    {
        'regex': (exclude_markdown_table(r'[\u0591-\u05C7-[\u05be]]'), ''),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/nikud/',
        'info': '05be it upper dash which we want to keep'
    },
    {
        'regex': (exclude_markdown_table(r'\u00A0'), ' '),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/nikud/',
        'info': ''
    },
    {
        'regex': (exclude_markdown_table(r'[\u200E\u200F\u202A-\u202F]'), ''),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/nikud/',
        'info': ''
    },
    {
        'regex': (r'</?MARKDOWN_TABLE>', ''),
        'bucket_name': 'gepeta-datasets',
        'path': f'partly-processed/{CLEAN_PATH}/markdown/',
        'info': 'Remove MARKDOWN_TABLE opening and closing tags'
    }

]


