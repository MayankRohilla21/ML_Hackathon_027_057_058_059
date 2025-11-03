# src/train_rl.py
import os
import time
import argparse
import numpy as np

from hangman_env import HangmanEnv
from dqn_agent import DQNAgent

"""
Training script for DQN agent on HangmanEnv.
... (docstring) ...
"""

def encode_observation(obs, max_len=12):
    # obs["letter_priors"] -> (26,)
    # obs["guessed"] -> list of guessed letters
    # obs["masked"] -> string length L <= max_len
    lp = np.array(obs["letter_priors"], dtype=np.float32)  # (26,)
    guessed = np.zeros(26, dtype=np.float32)
    for ch in obs["guessed"]:
        guessed[ord(ch) - ord('a')] = 1.0

    masked = obs["masked"]
    L = len(masked)
    # positions: for pos 0..max_len-1, create 27-dim onehot (0..25 letter, index 26 = blank/unused)
    pos_vecs = np.zeros((max_len, 27), dtype=np.float32)
    for i in range(max_len):
        if i >= L:
            pos_vecs[i, 26] = 1.0  # unused
        else:
            ch = masked[i]
            if ch == "_":
                pos_vecs[i, 26] = 1.0
            else:
                pos_vecs[i, ord(ch) - ord('a')] = 1.0
    pos_flat = pos_vecs.flatten()
    return np.concatenate([lp, guessed, pos_flat], axis=0)

def train(
    episodes=20000,
    max_len=12,
    max_lives=6,
    save_path="models/dqn_agent.pth",
    progress_every=500,
    seed=0
):
    # --- THIS IS THE MODIFIED BLOCK ---
    # We now explicitly pass the file path for the *training* data
    env = HangmanEnv(
        word_file_path="data/corpus.txt", 
        min_len=3, 
        max_len=max_len, 
        max_lives=max_lives, 
        seed=seed
    )
    # --- END OF MODIFICATION ---

    # if you trained HMM and want to attach it for priors, load and set it here:
    # from hmm_train import HiddenMarkovModel  # <-- Fixed import path
    # hmm = HiddenMarkovModel.load("data/hmm.jsonl") # <-- Fixed file path
    # env.set_hmm(hmm)

    sample_obs = env.reset()
    state_dim = encode_observation(sample_obs, max_len=max_len).shape[0]

    agent = DQNAgent(
        input_dim=state_dim,
        action_dim=26,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=15000,
        buffer_capacity=200000,
        batch_size=64,
        target_update_freq=1000
    )

    rewards_window = []
    wins_window = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs = env.reset()
        state = encode_observation(obs, max_len=max_len)
        total_reward = 0.0
        steps = 0
        while True:
            legal_idxs = env.legal_actions()  # list of indices allowed
            legal_mask = np.zeros(26, dtype=bool) # <-- Fixed deprecated np.bool_
            legal_mask[legal_idxs] = True

            action = agent.act(state, legal_mask=legal_mask)
            next_obs, reward, done, info = env.step(action)
            next_state = encode_observation(next_obs, max_len=max_len)
            agent.store(state, action, reward, next_state, float(done))
            loss = agent.update()

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                win = ("terminal" in info and info.get("terminal") == "win") or ("_" not in next_obs["masked"])
                wins_window.append(1.0 if win else 0.0)
                rewards_window.append(total_reward)
                break

        # Logging
        if ep % progress_every == 0:
            avg_reward = np.mean(rewards_window[-progress_every:]) if rewards_window else 0.0
            win_rate = np.mean(wins_window[-progress_every:]) if wins_window else 0.0
            print(f"Episode {ep}/{episodes} | avg_reward(last {progress_every}): {avg_reward:.3f} | win_rate: {win_rate:.3f} | buffer: {len(agent.replay)} | eps: {agent._epsilon():.3f} | time_elapsed: {time.time()-start_time:.1f}s")
            # save checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)

    # final save
    agent.save(save_path)
    print("Training finished. Model saved to", save_path)
    return agent, env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max_len", type=int, default=12)
    parser.add_argument("--max_lives", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="models/dqn_agent.pth")
    parser.add_argument("--progress_every", type=int, default=500)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        max_len=args.max_len,
        max_lives=args.max_lives,
        save_path=args.save_path,
        progress_every=args.progress_every
    )