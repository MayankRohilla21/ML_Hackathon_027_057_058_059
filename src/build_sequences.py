# src/build_sequences.py
import os
import json
from src.utils import load_wordlist, tokenize_word, CORPUS_PATH

def main(out_path="data/sequences.jsonl", min_len=3, max_len=12):
    """
    Convert words from corpus into character sequences
    and save them in JSON Lines format.
    Each line looks like:
      {"word": "apple", "chars": ["a", "p", "p", "l", "e"]}
    """
    corpus = CORPUS_PATH
    if not os.path.exists(corpus):
        raise FileNotFoundError(f"Corpus not found at {corpus}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for w in load_wordlist(corpus, min_len=min_len, max_len=max_len):
            seq = tokenize_word(w)
            fout.write(json.dumps({"word": w, "chars": seq}) + "\n")
            count += 1

    print(f"Wrote {count} sequences to {out_path}")

if __name__ == "__main__":
    main()
