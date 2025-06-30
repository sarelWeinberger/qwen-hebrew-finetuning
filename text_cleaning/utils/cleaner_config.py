from typing import Dict, List
from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaners.regex_cleaner import RegExCleaner
from cleaners.composite_cleaner import CompositeCleaner
from utils.cleaner_constants import SOURCES, CLEANUP_RULES

def create_cleaner_registries(debug_mode: bool = False):
    """
    Create registry dictionaries for cleaners.
    
    Args:
        debug_mode: Whether to enable debug mode for regex cleaner
    """
    # Extract regex patterns from CLEANUP_RULES
    regex_patterns = [rule['regex'] for rule in CLEANUP_RULES]
    
    # Create registry dictionaries
    duplicate_cleaner_registry: Dict[str, DuplicateRemoverCleaner] = {
        source: DuplicateRemoverCleaner() for source in SOURCES
    }

    regex_cleaner_registry: Dict[str, RegExCleaner] = {
        source: RegExCleaner(patterns=regex_patterns, debug_mode=debug_mode) for source in SOURCES
    }

    # Create composite cleaners for each source
    composite_cleaner_registry: Dict[str, CompositeCleaner] = {
        source: CompositeCleaner([duplicate_cleaner_registry[source], regex_cleaner_registry[source]])
        for source in SOURCES
    }

    # Create lists of cleaners for each type
    duplicate_cleaners: List[DuplicateRemoverCleaner] = list(duplicate_cleaner_registry.values())
    regex_cleaners: List[RegExCleaner] = list(regex_cleaner_registry.values())
    composite_cleaners: List[CompositeCleaner] = list(composite_cleaner_registry.values())

    # Create zipped lists of cleaners and sources
    cleaner_source_pairs = list(zip(composite_cleaners, SOURCES))
    
    return {
        'duplicate_cleaner_registry': duplicate_cleaner_registry,
        'regex_cleaner_registry': regex_cleaner_registry,
        'composite_cleaner_registry': composite_cleaner_registry,
        'duplicate_cleaners': duplicate_cleaners,
        'regex_cleaners': regex_cleaners,
        'composite_cleaners': composite_cleaners,
        'cleaner_source_pairs': cleaner_source_pairs
    }

# Create default registries (non-debug mode)
default_registries = create_cleaner_registries(debug_mode=False)

# Export the default registries for backward compatibility
duplicate_cleaner_registry = default_registries['duplicate_cleaner_registry']
regex_cleaner_registry = default_registries['regex_cleaner_registry']
composite_cleaner_registry = default_registries['composite_cleaner_registry']
duplicate_cleaners = default_registries['duplicate_cleaners']
regex_cleaners = default_registries['regex_cleaners']
composite_cleaners = default_registries['composite_cleaners']
cleaner_source_pairs = default_registries['cleaner_source_pairs'] 