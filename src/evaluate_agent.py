import argparse
import numpy as np
from hangman_env import HangmanEnv
from dqn_agent import DQNAgent

# We reimplement encode_observation here to avoid circular imports if necessary.
def encode_observation_local(obs, max_len=12):
    lp = np.array(obs["letter_priors"], dtype=np.float32)
    guessed = np.zeros(26, dtype=np.float32)
    for ch in obs["guessed"]:
        guessed[ord(ch) - ord('a')] = 1.0
    masked = obs["masked"]
    L = len(masked)
    pos_vecs = np.zeros((max_len, 27), dtype=np.float32)
    for i in range(max_len):
        if i >= L:
            pos_vecs[i, 26] = 1.0
        else:
            ch = masked[i]
            if ch == "_":
                pos_vecs[i, 26] = 1.0
            else:
                pos_vecs[i, ord(ch) - ord('a')] = 1.0
    pos_flat = pos_vecs.flatten()
    return np.concatenate([lp, guessed, pos_flat], axis=0)

def evaluate(model_path="models/dqn_agent.pth", episodes=1000, max_len=12, seed=42):
    
    # --- THIS IS THE MODIFIED BLOCK ---
    # We now explicitly pass the file path for the test data
    env = HangmanEnv(
        word_file_path="data/test.txt", 
        min_len=3, 
        max_len=max_len, 
        seed=seed
    )
    # --- END OF MODIFICATION ---
    
    # create a sample obs to determine input dim
    obs = env.reset()
    state_dim = encode_observation_local(obs, max_len=max_len).shape[0]

    # <-- This constructor call is now correct -->
    agent = DQNAgent(input_dim=state_dim) 
    
    print(f"Loading model from {model_path}...")
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'python src/train_rl.py' first to train and save the model.")
        return # Exit the function
        
    agent.model.eval()

    wins = 0
    total_steps = []
    total_rewards = []

    # --- Added counters for wrong and repeated guesses ---
    total_wrong_guesses = 0
    total_repeated_guesses = 0

    print(f"Running evaluation for {episodes} episodes...")
    for ep in range(episodes):
        obs = env.reset()
        state = encode_observation_local(obs, max_len=max_len)
        done = False
        steps = 0
        ep_reward = 0.0
        while not done:
            legal_idxs = env.legal_actions()
            legal_mask = np.zeros(26, dtype=bool)  # Fixed deprecated np.bool_
            legal_mask[legal_idxs] = True
            a = agent.act(state, legal_mask=legal_mask)
            next_obs, rew, done, info = env.step(a)

            # --- NEW: Count wrong & repeated guesses ---
            reason = info.get("reason", "")
            if reason == "wrong":
                total_wrong_guesses += 1
            elif reason == "repeated_guess":
                total_repeated_guesses += 1

            state = encode_observation_local(next_obs, max_len=max_len)
            steps += 1
            ep_reward += rew
        win = "_" not in next_obs["masked"]
        if win:
            wins += 1
        total_steps.append(steps)
        total_rewards.append(ep_reward)

    win_rate = wins / episodes

    # --- NEW: Compute Final Score as per Hackathon formula ---
    final_score = (win_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Evaluated {episodes} episodes")
    print(f"Win rate (Success Rate): {win_rate:.3f}")
    print(f"Total Wrong Guesses: {total_wrong_guesses}")
    print(f"Total Repeated Guesses: {total_repeated_guesses}")
    print(f"Final Score: {final_score:.2f}")
    print(f"Avg steps per episode: {np.mean(total_steps):.2f}")
    print(f"Avg reward per episode: {np.mean(total_rewards):.3f}")

# --- Fixed main guard syntax ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/dqn_agent.pth")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=12)
    args = parser.parse_args()

    evaluate(model_path=args.model_path, episodes=args.episodes, max_len=args.max_len)