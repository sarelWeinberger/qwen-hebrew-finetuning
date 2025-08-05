#!/usr/bin/env python3
"""
Test script to verify that regex patterns exclude content within MARKDOWN_TABLE tags.
"""

import regex
import pandas as pd
from utils.cleaner_constants import CLEANUP_RULES

def test_markdown_table_exclusion():
    """Test that regex patterns don't match content within MARKDOWN_TABLE tags."""
    
    # Test cases
    test_cases = [
        # Case 1: Text outside MARKDOWN_TABLE should be processed
        {
            'text': 'This text has &quot;quotes&quot; and &#39;apostrophes&#39;',
            'expected_processed': True,
            'description': 'Text outside MARKDOWN_TABLE should be processed'
        },
        # Case 2: Text inside MARKDOWN_TABLE should NOT be processed
        {
            'text': '<MARKDOWN_TABLE>This text has &quot;quotes&quot; and &#39;apostrophes&#39;</MARKDOWN_TABLE>',
            'expected_processed': False,
            'description': 'Text inside MARKDOWN_TABLE should NOT be processed'
        },
        # Case 3: Mixed content - only outside should be processed
        {
            'text': 'Outside text &quot;quotes&quot; <MARKDOWN_TABLE>Inside &quot;quotes&quot;</MARKDOWN_TABLE> More outside &#39;apostrophes&#39;',
            'expected_processed': True,
            'description': 'Mixed content - only outside should be processed'
        },
        # Case 4: MARKDOWN_TABLE tags should be removed (last regex)
        {
            'text': '<MARKDOWN_TABLE>Content</MARKDOWN_TABLE>',
            'expected_processed': True,
            'description': 'MARKDOWN_TABLE tags should be removed'
        }
    ]
    
    print("Testing MARKDOWN_TABLE exclusion...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Input: {test_case['text']}")
        
        # Apply all regex rules except the last one (MARKDOWN_TABLE removal)
        processed_text = test_case['text']
        changes_made = False
        
        for rule in CLEANUP_RULES[:-1]:  # Exclude the last rule (MARKDOWN_TABLE removal)
            pattern = rule['regex'][0]
            replacement = rule['regex'][1]
            
            # Apply the regex
            new_text, count = regex.subn(pattern, replacement, processed_text)
            if count > 0:
                changes_made = True
                processed_text = new_text
                print(f"  Applied: {rule['info'][:50]}...")
        
        print(f"Output: {processed_text}")
        print(f"Changes made: {changes_made}")
        
        # Check if the result matches expectations
        if test_case['expected_processed']:
            if changes_made:
                print("✅ PASS: Expected processing occurred")
            else:
                print("❌ FAIL: Expected processing but none occurred")
        else:
            if not changes_made:
                print("✅ PASS: Expected no processing and none occurred")
            else:
                print("❌ FAIL: Expected no processing but changes were made")
    
    # Test the last rule (MARKDOWN_TABLE removal) separately
    print(f"\nTest Case {len(test_cases) + 1}: MARKDOWN_TABLE tag removal")
    test_text = '<MARKDOWN_TABLE>Content</MARKDOWN_TABLE>'
    print(f"Input: {test_text}")
    
    last_rule = CLEANUP_RULES[-1]
    pattern = last_rule['regex'][0]
    replacement = last_rule['regex'][1]
    
    processed_text, count = regex.subn(pattern, replacement, test_text)
    print(f"Output: {processed_text}")
    print(f"Changes made: {count > 0}")
    
    if processed_text == 'Content':
        print("✅ PASS: MARKDOWN_TABLE tags were removed")
    else:
        print("❌ FAIL: MARKDOWN_TABLE tags were not removed properly")

def test_specific_patterns():
    """Test specific regex patterns to ensure they work correctly."""
    
    print("\n" + "=" * 60)
    print("Testing Specific Patterns")
    print("=" * 60)
    
    # Test HTML escape codes
    test_texts = [
        'Outside &quot;quotes&quot; <MARKDOWN_TABLE>Inside &quot;quotes&quot;</MARKDOWN_TABLE>',
        'Outside &#39;apostrophes&#39; <MARKDOWN_TABLE>Inside &#39;apostrophes&#39;</MARKDOWN_TABLE>',
        'Multiple &quot;quotes&quot; and &#39;apostrophes&#39; <MARKDOWN_TABLE>Protected &quot;quotes&quot;</MARKDOWN_TABLE>'
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        
        # Apply HTML escape code rules
        for rule in CLEANUP_RULES[:2]:  # First two rules are HTML escape codes
            pattern = rule['regex'][0]
            replacement = rule['regex'][1]
            
            new_text, count = regex.subn(pattern, replacement, text)
            if count > 0:
                print(f"  Applied: {rule['info'][:30]}...")
                print(f"  Result: {new_text}")
                text = new_text

if __name__ == "__main__":
    test_markdown_table_exclusion()
    test_specific_patterns() 