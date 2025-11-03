import torch
import numpy as np
from hangman_env import HangmanEnv
# Import the agent AND the network definition
from train_dqn import DQNAgent, DQN_TwoTower
import time

def load_test_words(filepath="test.txt"):
    """Loads the hidden test set of words."""
    words = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip().upper()
                if word.isalpha():
                    words.append(word)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    return words

def evaluate_agent():
    print("--- 🧠 Starting Final Evaluation ---")
    
    # --- 1. Load Environment and Test Words ---
    env = HangmanEnv(oracle_path="hmm_oracles.pkl", corpus_path="corpus.txt")
    test_words = load_test_words()
    
    if not test_words:
        print("Cannot evaluate. No test words found.")
        return
        
    n_games = len(test_words)
    print(f"Loaded {n_games} words from test.txt.")
    
    # --- 2. Load the Trained Agent ---
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    stats_size = env.stats_size
    word_len = env.max_word_len
    word_encoding_size = env.letter_encoding_size
    
    # Initialize the agent
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        stats_size=stats_size,
        word_len=word_len, 
        word_encoding_size=word_encoding_size
    )
    
    # Load the saved model weights
    try:
        agent.qnetwork_local.load_state_dict(torch.load("dqn_agent_final.pth"))
    except FileNotFoundError:
        print("Error: dqn_agent_final.pth not found.")
        print("Did the training script finish?")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you change the network structure in train_dqn.py but not here?")
        return
        
    agent.qnetwork_local.eval()
    print(f"Successfully loaded model: dqn_agent_final.pth")

    # --- 3. Run the Evaluation ---
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    
    start_time = time.time()
    
    for i in range(n_games):
        secret_word = test_words[i]
        
        # --- Manual Reset for Test Word ---
        word_len_current = len(secret_word)
        if word_len_current not in env.hmm_oracles:
            print(f"Warning: No HMM for length {word_len_current} (word: {secret_word}). Skipping.")
            continue
            
        env.secret_word = secret_word
        model = env.hmm_oracles[word_len_current]
        
        log_floor = -1e10 
        with np.errstate(divide='ignore'):
            env.log_startprob = np.log(model.startprob_)
            env.log_transmat = np.log(model.transmat_)
            env.log_emissionprob = np.log(model.emissionprob_)
        env.log_startprob[env.log_startprob == -np.inf] = log_floor
        env.log_transmat[env.log_transmat == -np.inf] = log_floor
        env.log_emissionprob[env.log_emissionprob == -np.inf] = log_floor
        env.n_components = model.n_components
        
        env.masked_word = "_" * word_len_current
        env.lives_left = env.max_lives
        env.guessed_letters = set()
        
        state = env._get_state()
        # --- End Manual Reset ---
        
        done = False
        game_wrong_guesses = 0
        
        while not done:
            # --- Get Action (Epsilon = 0) ---
            action = agent.act(state, eps=0.0) # Pure exploitation
            
            # --- Take Step ---
            next_state, reward, done, _, info = env.step(action)
            
            if info.get('msg') == "Wrong guess":
                game_wrong_guesses += 1
            # Repeated guesses are already handled by our agent's act() mask
            
            state = next_state
        
        if info.get('msg') == "Game won!":
            total_wins += 1
            
        total_wrong_guesses += game_wrong_guesses
        
        if (i+1) % 200 == 0:
            print(f"Played {i+1}/{n_games} games...")
            
    end_time = time.time()
    
    # --- 4. Calculate Final Score (with correct formula) ---
    print("\n--- 🏁 Final Evaluation Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total Games Played: {n_games}")
    
    success_rate_decimal = total_wins / n_games
    success_rate_percent = success_rate_decimal * 100
    
    print(f"Total Wins: {total_wins} ({success_rate_percent:.2f}%)")
    print(f"Total Wrong Guesses: {total_wrong_guesses}")
    print(f"Total Repeated Guesses: {total_repeated_guesses}") # Should be 0

    score_multiplier = n_games
    if n_games == 1000 or n_games == 2000:
        final_score = (success_rate_percent * score_multiplier) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        print("\n--- 🏆 FINAL SCORE ---")
        print(f"Calculation: ({success_rate_percent:.2f} * {score_multiplier}) - ({total_wrong_guesses} * 5) - ({total_repeated_guesses} * 2)")
        print(f"Final Score: {final_score:.2f}")
    else:
        print(f"Score (non-standard test size): {final_score:.2f}")

if __name__ == "__main__":
    evaluate_agent()
