import numpy as np
from hangman_env import HangmanEnv  # We only need the environment
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

def evaluate_baseline_agent():
    print("--- 🔬 Starting HMM 'Oracle' Baseline Test ---")
    
    # --- 1. Load Environment and Test Words ---
    env = HangmanEnv(oracle_path="hmm_oracles.pkl", corpus_path="corpus.txt")
    test_words = load_test_words()
    
    if not test_words:
        print("Cannot evaluate. No test words found.")
        return
        
    n_games = len(test_words)
    print(f"Loaded {n_games} words from test.txt.")
    print("Agent Strategy: Greedy HMM (No RL, No Epsilon)")
    
    # --- 2. Run the Evaluation ---
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0 # Should be 0, but we check
    
    start_time = time.time()
    
    for i in range(n_games):
        secret_word = test_words[i]
        
        # --- Manual Reset for Test Word ---
        word_len = len(secret_word)
        if word_len not in env.hmm_oracles:
            print(f"Warning: No HMM for length {word_len} (word: {secret_word}). Skipping.")
            continue
            
        env.secret_word = secret_word
        model = env.hmm_oracles[word_len]
        
        # Manually set HMM params (from env.reset())
        log_floor = -1e10 
        with np.errstate(divide='ignore'):
            env.log_startprob = np.log(model.startprob_)
            env.log_transmat = np.log(model.transmat_)
            env.log_emissionprob = np.log(model.emissionprob_)
        env.log_startprob[env.log_startprob == -np.inf] = log_floor
        env.log_transmat[env.log_transmat == -np.inf] = log_floor
        env.log_emissionprob[env.log_emissionprob == -np.inf] = log_floor
        env.n_components = model.n_components
        # End HMM param set
        
        env.masked_word = "_" * word_len
        env.lives_left = env.max_lives
        env.guessed_letters = set()
        
        state = env._get_state()
        # --- End Manual Reset ---
        
        done = False
        game_wrong_guesses = 0
        
        while not done:
            # --- This is the Baseline "Brain" ---
            
            # 1. Get the Oracle's advice (HMM probs) from the state
            # state[0] = lives
            # state[1:27] = guessed_mask
            # state[27:53] = hmm_probs
            hmm_probs = state[27:53]
            
            # 2. Get the guessed_mask
            guessed_mask = state[1:27]
            
            # 3. Apply the mask: set prob of guessed letters to -1
            hmm_probs[guessed_mask == 1] = -1.0
            
            # 4. Choose the action with the highest probability
            action = np.argmax(hmm_probs)
            # --- End Baseline "Brain" ---

            # --- Take Step ---
            next_state, reward, done, _, info = env.step(action)
            
            if info.get('msg') == "Wrong guess":
                game_wrong_guesses += 1
            
            state = next_state
        
        # --- Game Finished ---
        if info.get('msg') == "Game won!":
            total_wins += 1
            
        total_wrong_guesses += game_wrong_guesses
        
        if (i+1) % 200 == 0:
            print(f"Played {i+1}/{n_games} games...")
            
    end_time = time.time()
    
    # --- 3. Calculate Final Score ---
    print("\n--- 🏁 Baseline Test Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total Games Played: {n_games}")
    
    # --- *** FIX: Calculate both decimal and percent *** ---
    success_rate_decimal = total_wins / n_games
    success_rate_percent = success_rate_decimal * 100  # e.g., 30.7
    # --- *** END FIX *** ---

    print(f"\n--- HMM Oracle Performance ---")
    print(f"Total Wins: {total_wins} ({success_rate_percent:.2f}%)")
    print(f"Total Wrong Guesses: {total_wrong_guesses}")

    # Note: The PDF is ambiguous, saying 1000 games but the test set is 2000.
    # We will assume the multiplier in the formula is the number of games played.
    score_multiplier = n_games

    if n_games == 1000 or n_games == 2000:
        # --- *** FIX: Use success_rate_percent in the formula *** ---
        final_score = (success_rate_percent * score_multiplier) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        
        print("\n--- 🏆 BASELINE FINAL SCORE ---")
        print(f"Calculation: ({success_rate_percent:.2f} * {score_multiplier}) - ({total_wrong_guesses} * 5) - (0 * 2)")
        print(f"Final Score: {final_score:.2f}")
        # --- *** END FIX *** ---
        
    else:
        print("\n--- 🏆 FINAL SCORE (Formula mismatch) ---")
        print("Your test set was not 1000 or 2000 words.")
        print("Please report the stats above (Wins, Wrong Guesses) manually.")

if __name__ == "__main__":
    evaluate_baseline_agent()