# src/hmm_eval.py
import json
import random
import numpy as np
from hmm_train import HiddenMarkovModel  # <-- FIXED IMPORT
from utils import load_wordlist, CORPUS_PATH # <-- ENSURED IMPORT IS CORRECT

def main(test_path="data/test.txt", sample_size=2000):
    """Evaluate the trained HMM model on unseen words."""
    # Load trained HMM
    hmm = HiddenMarkovModel.load("data/hmm.jsonl")

    # Load test corpus (unseen words)
    words = list(load_wordlist(test_path, min_len=3, max_len=12))
    random.shuffle(words)
    test_words = words[:sample_size]
    
    if not test_words:
        print(f"Error: No data loaded from {test_path}. Cannot evaluate.")
        return

    # Compute log-likelihood for each word
    scores = [hmm.log_likelihood(list(w)) for w in test_words]
    avg_log = np.mean(scores)

    # Display results
    print(f"Evaluated {len(test_words)} unseen words")
    print(f"Average log-likelihood: {avg_log:.4f}\n")
    
    print("Sample evaluations:")
    for w in test_words[:10]:
        print(f"{w:10s} -> {hmm.log_likelihood(list(w)):.4f}")

if __name__ == "__main__":
    main()