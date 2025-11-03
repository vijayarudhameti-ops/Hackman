from collections import defaultdict
import re

def load_and_group_corpus(filepath="corpus.txt"):
    """
    Loads words from the corpus, cleans them, and groups them by length.
    """
    # Use a defaultdict to automatically handle new list creation
    words_by_length = defaultdict(list)
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Clean the word: strip whitespace, convert to uppercase
                # We also filter for only alphabetic words, just in case
                word = line.strip().upper()
                if word.isalpha():
                    words_by_length[len(word)].append(word)
                    
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    
    print(f"Corpus loaded. Found words for {len(words_by_length)} different lengths.")
    # Example: Print how many 5-letter words were found
    if 5 in words_by_length:
        print(f"Example: Found {len(words_by_length[5])} 5-letter words.")
        
    return words_by_length

# --- Run this first ---
# This dictionary is the foundation for training your HMMs
grouped_words = load_and_group_corpus("corpus.txt")