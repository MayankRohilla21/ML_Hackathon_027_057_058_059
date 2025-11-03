# src/hangman_env.py
import random
import numpy as np
from collections import Counter
from utils import load_wordlist
from hmm_train import HiddenMarkovModel

ALPHABET = [chr(i) for i in range(ord("a"), ord("z") + 1)]


class HangmanEnv:
    """
    Simple Hangman environment suitable for RL.
    ... (docstring) ...
    """

    # --- THIS IS THE MODIFIED __init__ ---
    def __init__(self, word_file_path="data/corpus.txt", min_len=3, max_len=12, max_lives=6, seed=None):
        """
        word_file_path: path to the word list to sample from
        max_lives: number of allowed incorrect guesses
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # <-- MODIFIED LINE: Uses the word_file_path argument -->
        self.words = list(load_wordlist(file_path=word_file_path, min_len=min_len, max_len=max_len))

        if not self.words:
            raise ValueError(f"No words available from {word_file_path}. Check file / length filters.")

        self.max_lives = max_lives
        self.hmm = None
    # --- END OF MODIFICATION ---

        # runtime state
        self.target_word = None
        self.masked = None
        self.guessed = None
        self.remaining_lives = None
        self.done = None
        self.last_reward = 0.0
        self.candidate_words = None

    # -----------------------
    # Optional: attach HMM
    # -----------------------
    def set_hmm(self, hmm: HiddenMarkovModel):
        """Attach a trained HiddenMarkovModel to the environment for priors."""
        self.hmm = hmm

    # -----------------------
    # Helper utilities
    # -----------------------
    @staticmethod
    def letter_to_index(letter: str) -> int:
        return ord(letter) - ord("a")

    @staticmethod
    def index_to_letter(idx: int) -> str:
        return ALPHABET[idx]

    def _init_candidate_words(self):
        """Initialize candidate words matching the current mask (used when HMM is available)."""
        L = len(self.target_word)
        self.candidate_words = [w for w in self.words if len(w) == L]
        # filter by already revealed letters
        self.candidate_words = [
            w for w in self.candidate_words
            if all(self.masked[i] == "_" or self.masked[i] == w[i] for i in range(L))
        ]

    # -----------------------
    # Environment API
    # -----------------------
    def reset(self, word: str = None):
        """
        Start a new episode.
        If `word` is provided, use it (for testing/evaluation), otherwise sample randomly.
        Returns initial observation (dictionary).
        """
        if word is None:
            self.target_word = random.choice(self.words)
        else:
            if word not in self.words:
                # still allow words outside list (but warn); we keep it allowed
                self.target_word = word
            else:
                self.target_word = word

        self.masked = ["_"] * len(self.target_word)
        self.guessed = set()
        self.remaining_lives = self.max_lives
        self.done = False
        self.last_reward = 0.0
        self._init_candidate_words()

        return self._get_observation()

    def _get_observation(self):
        """
        Returns a dictionary observation:
          - masked (string): current masked word (e.g., '_a__e')
          - guessed (list): sorted guessed letters
          - remaining_lives (int)
          - letter_priors (np.array of shape (26,)) optional: HMM prior probs
        """
        masked_str = "".join(self.masked)
        guessed_sorted = sorted(list(self.guessed))

        # compute HMM priors if available
        if self.hmm is not None:
            # try to derive candidate words consistent with mask and use them if possible
            self._init_candidate_words()
            priors = self.hmm.get_letter_scores_for_mask(masked_str, candidate_words=self.candidate_words)
            # ensure ordering a..z
            letter_priors = np.array([priors.get(c, 0.0) for c in ALPHABET], dtype=np.float32)
            # normalize to sum 1 (if all zeros, fallback to uniform)
            s = letter_priors.sum()
            if s > 0:
                letter_priors /= s
            else:
                letter_priors = np.ones(26, dtype=np.float32) / 26.0
        else:
            # uniform priors if no HMM attached
            letter_priors = np.ones(26, dtype=np.float32) / 26.0

        return {
            "masked": masked_str,
            "guessed": guessed_sorted,
            "remaining_lives": self.remaining_lives,
            "letter_priors": letter_priors,
        }

    def legal_actions(self):
        """Return list of ints (indices) corresponding to letters not yet guessed."""
        return [self.letter_to_index(l) for l in ALPHABET if l not in self.guessed]

    def step(self, action: int):
        """
        Apply one action (letter index).
        Returns: observation, reward (float), done (bool), info (dict)
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new one.")

        if action < 0 or action >= 26:
            raise ValueError("Action must be index 0..25 representing letter a..z")

        letter = self.index_to_letter(action)

        # repeated guess
        if letter in self.guessed:
            reward = -0.2
            self.last_reward = reward
            info = {"reason": "repeated_guess", "letter": letter}
            # no change to masked or lives
            return self._get_observation(), reward, self.done, info

        # register guess
        self.guessed.add(letter)

        # correct guess?
        if letter in self.target_word:
            revealed = 0
            for i, ch in enumerate(self.target_word):
                if ch == letter and self.masked[i] == "_":
                    self.masked[i] = letter
                    revealed += 1
            # reward proportional to new letters revealed (encourages multi-letter reveals)
            reward = 1.0 * revealed if revealed > 0 else 0.0
            self.last_reward = reward
            info = {"reason": "correct", "letter": letter, "revealed": revealed}

            # update candidates
            self._init_candidate_words()

            # check win
            if "_" not in self.masked:
                self.done = True
                reward += 10.0  # final win bonus
                self.last_reward = reward
                info["terminal"] = "win"
            return self._get_observation(), reward, self.done, info

        else:
            # wrong guess
            self.remaining_lives -= 1
            reward = -0.5
            self.last_reward = reward
            info = {"reason": "wrong", "letter": letter, "remaining_lives": self.remaining_lives}
            # check lose
            if self.remaining_lives <= 0:
                self.done = True
                reward += -10.0  # final lose penalty
                self.last_reward = reward
                info["terminal"] = "lose"
                # reveal word in observation
                self.masked = list(self.target_word)
            return self._get_observation(), reward, self.done, info

    def render(self, verbose=True):
        """Simple text render returning the current state as string (and optionally prints)."""
        s = f"Word: {' '.join(self.masked)} | Guessed: {''.join(sorted(self.guessed))} | Lives: {self.remaining_lives}"
        if verbose:
            print(s)
        return s

    # -----------------------
    # Convenience: evaluate an oracle policy (for debugging)
    # -----------------------
    def play_random(self, max_steps=100):
        """Play until done by choosing random legal actions. Returns final reward and success (True/False)."""
        obs = self.reset()
        steps = 0
        while not self.done and steps < max_steps:
            actions = self.legal_actions()
            a = random.choice(actions)
            obs, r, done, info = self.step(a)
            steps += 1
        success = "_" not in self.masked
        return success, self.last_reward, steps