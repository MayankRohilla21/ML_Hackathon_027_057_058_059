# ML_Hackathon_027_057_058_059
ML_Hackathon_AM_027_057_058_059  Simulating hangman game using Hidden Markov Model (HMMs) and Reinforcement Learning

ML-Hackman: A Hybrid AI Hangman Solver
This project uses a hybrid AI approach to master the game of Hangman. It combines a statistical Hidden Markov Model (HMM) to understand word structure with a Deep Q-Network (DQN) to learn and execute a winning game strategy.

The HMM acts as a "Bookworm," providing statistical hints (letter probabilities), while the DQN acts as the "Player," learning a strategic policy to win the game.

🧠 The Approach: A Two-Brain AI
Our method is a hybrid model that uses the strengths of two different AI techniques.

1. The (Hidden Markov Model)
What it is: A statistical model trained on data/corpus.txt using the src/hmm_train.py script.
Its Job: It learns the statistical patterns of English words (e.g., "E" is common, "Q" is usually followed by "U").
Its Role: It doesn't play the game. Instead, it acts as an advisor to the Player. It provides a "hint" in the form of letter_priors (a probability for each letter) based on the current state of the board.
2. The "Player" (Deep Q-Network)
What it is: A Deep Reinforcement Learning agent trained by src/train_rl.py. This is the "brain" that actually plays the game.
Its Job: To learn a winning strategy (a "policy") by playing thousands of games in the hangman_env.py environment.
How it Learns:
Hybrid State: The "state" it sees includes the game board, the guessed letters, AND the statistical hints from the HMM.
Trial and Error: It learns by receiving rewards (e.g., +1 for a correct letter, +10 for a win) and penalties (e.g., -0.5 for a wrong guess, -10 for a loss).
Replay Buffer: It uses a "memory" to store all past experiences and learns by studying random batches of them.
3. The "Final Exam" (Fair Evaluation)
data/corpus.txt (The Homework): Used only for training the HMM and the DQN agent.
data/test.txt (The Exam): An unseen list of words used only by src/evaluate_agent.py to get a true, unbiased score of the agent's performance.


⚙ Setup and Installation
Clone the repository:

git clone <your-repo-url>
cd ML_Hackman
Create and activate a virtual environment:

python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
Install the required packages: (You should create a requirements.txt file with the content below)

pip install -r requirements.txt
requirements.txt
🚀 How to Run
Run all commands from the root ML_Hackman directory (the one containing src and data).

1. Train the HMM
This reads data/corpus.txt and creates the data/hmm.jsonl hint file.

python src/hmm_train.py

(Optional) Evaluate the HMM
This checks your HMM's statistical model against the data/test.txt file.

Bash

python src/hmm_eval.py
Train the DQN Agent (The "Player")
This is the main training step. The agent will play thousands of games against data/corpus.txt and save its brain to models/dqn_agent.pth.

Bash

python src/train_rl.py --episodes 5000
(You can change 5000 to a larger number like 20000 for better, but slower, training.)

Run the Final Evaluation (The "Test")
This is the final step. It loads your trained agent and tests it against the unseen words in data/test.txt to get its true performance.

Bash

python src/evaluate_agent.py --episodes 2000
The Win Rate from this script is your project's final, true score.
