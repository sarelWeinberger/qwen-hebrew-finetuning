from typing import Dict, List
from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaners.regex_cleaner import RegExCleaner
from cleaners.composite_cleaner import CompositeCleaner
from utils.cleaner_constants import SOURCES, CLEANUP_RULES

# Create registry dictionaries
duplicate_cleaner_registry: Dict[str, DuplicateRemoverCleaner] = {
    source: DuplicateRemoverCleaner() for source in SOURCES
}

regex_cleaner_registry: Dict[str, RegExCleaner] = {
    source: RegExCleaner(patterns=CLEANUP_RULES) for source in SOURCES
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