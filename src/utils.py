# src/utils.py
import os

# Define paths for easy access
CORPUS_PATH = "data/corpus.txt"
TEST_PATH = "data/test.txt"

# <-- THIS FUNCTION IS MODIFIED to accept file_path -->
def load_wordlist(file_path=CORPUS_PATH, min_len=3, max_len=12, verbose=True):
    """
    Load a word list from the specified file, filtering by length.
    Defaults to CORPUS_PATH if no file_path is given.
    """
    if not os.path.exists(file_path):
        if verbose:
            print(f"Warning: Word file not found at {file_path}. Returning empty list.")
        return set()

    words = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.add(w)
    
    if verbose and not words:
        print(f"Warning: No words loaded from {file_path} matching criteria.")
        
    return words