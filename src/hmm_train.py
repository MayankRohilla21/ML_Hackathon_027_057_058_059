# src/hmm_train.py
import json
import os
import math
from collections import defaultdict, Counter
from utils import load_wordlist  # <-- ADDED THIS IMPORT

class HiddenMarkovModel:
    """
    A simple character-level Hidden Markov Model for Hangman word structure learning.
    It learns:
      - Transition probabilities between hidden states (START → MIDDLE → END)
      - Emission probabilities of letters from the MIDDLE state
    """

    def __init__(self, states=None, symbols=None, smoothing=1.0):
        # Define hidden states and observable symbols (a-z)
        self.states = states or ["START", "MIDDLE", "END"]
        self.symbols = symbols or [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.smoothing = smoothing

        # Store probabilities as nested Counters
        self.transitions = defaultdict(Counter)  # state_i → state_j
        self.emissions = defaultdict(Counter)    # state → letter

    # ---------------------------------------------------------
    def train(self, sequences):
        """
        Train the HMM using a list of sequences (each a list of characters).
        """
        for seq in sequences:
            if not seq:
                continue

            # START → MIDDLE → ... → END transitions
            self.transitions["START"]["MIDDLE"] += 1
            self.emissions["MIDDLE"][seq[0]] += 1

            # For each next character, remain in MIDDLE state
            for i in range(1, len(seq)):
                self.transitions["MIDDLE"]["MIDDLE"] += 1
                self.emissions["MIDDLE"][seq[i]] += 1

            # End transition
            self.transitions["MIDDLE"]["END"] += 1

        # Normalize all counts into probabilities
        self._normalize()

    # ---------------------------------------------------------
    def _normalize(self):
        """Convert raw counts into probability distributions."""
        # Transition probabilities
        for s in self.transitions:
            total = sum(self.transitions[s].values()) + self.smoothing * len(self.states)
            for s2 in self.states:
                self.transitions[s][s2] = (self.transitions[s][s2] + self.smoothing) / total

        # Emission probabilities
        for s in self.emissions:
            total = sum(self.emissions[s].values()) + self.smoothing * len(self.symbols)
            for sym in self.symbols:
                self.emissions[s][sym] = (self.emissions[s][sym] + self.smoothing) / total

    # ---------------------------------------------------------
    def emission_prob(self, state, symbol):
        """Return P(symbol | state)."""
        return self.emissions[state].get(symbol, 1e-8)

    def transition_prob(self, s1, s2):
        """Return P(next_state | current_state)."""
        return self.transitions[s1].get(s2, 1e-8)

    # ---------------------------------------------------------
    def log_likelihood(self, sequence):
        """
        Compute log-likelihood of a word (sequence of letters) under this model.
        Used for evaluation of unseen words.
        """
        if not sequence:
            return float("-inf")

        # --- THIS BLOCK IS THE FIXED LOGIC ---
        try:
            # P(START -> MIDDLE) * P(first_letter | MIDDLE)
            prob = self.transition_prob("START", "MIDDLE")
            prob *= self.emission_prob("MIDDLE", sequence[0])
        except IndexError:
            return float("-inf")

        # For the rest of the letters: P(MIDDLE -> MIDDLE) * P(letter | MIDDLE)
        for ch in sequence[1:]:
            prob *= self.transition_prob("MIDDLE", "MIDDLE")
            prob *= self.emission_prob("MIDDLE", ch)

        # Final transition: P(MIDDLE -> END)
        prob *= self.transition_prob("MIDDLE", "END")
        # --- END OF FIXED LOGIC ---

        return math.log(prob + 1e-12) # Add small epsilon to prevent log(0)


    # ---------------------------------------------------------
    def get_letter_scores_for_mask(self, masked_word, candidate_words=None):
        """
        Return probability distribution for letters given the current masked word.
        If candidate_words is provided, compute empirical probabilities from them.
        Otherwise, use the HMM's learned emission probabilities.
        """
        if candidate_words:
            L = len(masked_word)
            counts = Counter()
            total = 0
            for w in candidate_words:
                if len(w) != L:
                    continue
                # keep only words matching revealed letters
                if any((masked_word[i] != '_' and masked_word[i] != w[i]) for i in range(L)):
                    continue
                # count letters at blank positions
                for i in range(L):
                    if masked_word[i] == '_':
                        counts[w[i]] += 1
                        total += 1
            probs = {c: (counts.get(c, 0) / total if total > 0 else 1e-12) for c in self.symbols}
            return probs

        # fallback: use HMM emission distribution (MIDDLE state)
        return dict(self.emissions.get("MIDDLE", {}))

    # ---------------------------------------------------------
    def save(self, path="data/hmm.jsonl"):
        """Save trained HMM to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model = {
            "states": self.states,
            "symbols": self.symbols,
            "transitions": {s: dict(self.transitions[s]) for s in self.transitions},
            "emissions": {s: dict(self.emissions[s]) for s in self.emissions},
            "smoothing": self.smoothing
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        print(f"HMM model saved to {path}")

    # ---------------------------------------------------------
    @staticmethod
    def load(path="data/hmm.jsonl"):
        """Load an HMM model from a saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hmm = HiddenMarkovModel(
            states=data["states"],
            symbols=data["symbols"],
            smoothing=data.get("smoothing", 1.0)
        )
        hmm.transitions = defaultdict(Counter, {s: Counter(v) for s, v in data["transitions"].items()})
        hmm.emissions = defaultdict(Counter, {s: Counter(v) for s, v in data["emissions"].items()})
        print(f"HMM model loaded from {path}")
        return hmm


# ---------------------------------------------------------
# SCRIPT EXECUTION - ADDED THIS ENTIRE BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    
    print("Loading training data from 'data/corpus.txt'...")
    # Load the training words
    sequences = list(load_wordlist("data/corpus.txt", min_len=3, max_len=12))
    
    if not sequences:
        print("Error: No data loaded from 'data/corpus.txt'. File might be empty or in the wrong path.")
    else:
        print(f"Loaded {len(sequences)} words for training.")
        
        # 1. Create the model
        hmm = HiddenMarkovModel()
        
        # 2. Train the model
        print("Training HMM...")
        hmm.train(sequences)
        
        # 3. Save the model (this will create the file)
        print("Saving model...")
        hmm.save("data/hmm.jsonl")
        print("Training complete.")