import torch
import numpy as np
from hangman_env import HangmanEnv
from train_dqn import DQNAgent, DQN  # We re-use the Agent and DQN classes
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
    print("--- 🧠 Starting Evaluation ---")
    
    # --- 1. Load Environment and Test Words ---
    env = HangmanEnv(oracle_path="hmm_oracles.pkl", corpus_path="corpus.txt")
    test_words = load_test_words()
    
    if not test_words:
        print("Cannot evaluate. No test words found.")
        return
        
    # As per PDF, we evaluate on these words.
    # Note: Your PDF has a mismatch (1000 vs 2000 games).
    # We will run on ALL words found in test.txt.
    n_games = len(test_words) 
    print(f"Loaded {n_games} words from test.txt.")
    
    # --- 2. Load the Trained Agent ---
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the saved model weights
    try:
        agent.qnetwork_local.load_state_dict(torch.load("dqn_agent_final.pth"))
    except FileNotFoundError:
        print("Error: dqn_agent_final.pth not found.")
        print("Did the training script finish?")
        return
        
    # Set the network to "evaluation mode" (disables some training-only features)
    agent.qnetwork_local.eval()
    
    print(f"Successfully loaded model: dqn_agent_final.pth")

    # --- 3. Run the Evaluation ---
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    
    start_time = time.time()
    
    for i in range(n_games):
        # We must override the env's reset() to use the test words
        secret_word = test_words[i]
        
        # --- Manual Reset for Test Word ---
        word_len = len(secret_word)
        if word_len not in env.hmm_oracles:
            print(f"Warning: No HMM for length {word_len} (word: {secret_word}). Skipping.")
            continue
            
        env.secret_word = secret_word
        env.hmm_model = env.hmm_oracles[word_len]
        env.masked_word = "_" * word_len
        env.lives_left = env.max_lives
        env.guessed_letters = set()
        
        state = env._get_state()
        # --- End Manual Reset ---
        
        done = False
        game_wrong_guesses = 0
        game_repeated_guesses = 0
        
        while not done:
            # --- Get Action ---
            # We use Epsilon = 0.0 for pure exploitation (no random moves)
            # We MUST use the same action-masking logic
            
            guessed_mask = state[1:27]
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(env.device) # Use env's device
            
            with torch.no_grad():
                action_values = agent.qnetwork_local(state_tensor)
                
            # Apply action mask
            action_values[0][guessed_mask == 1] = -float('inf')
            action = np.argmax(action_values.cpu().data.numpy())
            
            # --- Check for Repeated Guess (for scoring) ---
            # The agent *shouldn't* do this, but we track it per the PDF
            if env.alphabet[action] in env.guessed_letters:
                game_repeated_guesses += 1
            
            # --- Take Step ---
            next_state, reward, done, _, info = env.step(action)
            
            if info.get('msg') == "Wrong guess":
                game_wrong_guesses += 1
            
            state = next_state
        
        # --- Game Finished ---
        if info.get('msg') == "Game won!":
            total_wins += 1
            
        total_wrong_guesses += game_wrong_guesses
        total_repeated_guesses += game_repeated_guesses
        
        if (i+1) % 200 == 0:
            print(f"Played {i+1}/{n_games} games...")
            
    end_time = time.time()
    
    # --- 4. Calculate Final Score ---
    print("\n--- 🏁 Evaluation Complete ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total Games Played: {n_games}")
    
    success_rate = total_wins / n_games
    
    print(f"Total Wins: {total_wins} ({success_rate * 100:.2f}%)")
    print(f"Total Wrong Guesses: {total_wrong_guesses}")
    print(f"Total Repeated Guesses: {total_repeated_guesses}")

    # --- Scoring Formula from PDF ---
    # Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
    # Note: The "2000" in the formula seems tied to playing 2000 games.
    # A more robust formula would be (Wins * 1000) if 1000 games
    # Let's use the provided formula *exactly*, assuming 2000 games
    
    if n_games == 1000 or n_games == 2000: # Handle the 1000/2000 mismatch
        score_multiplier = 2000 if n_games == 2000 else 1000 
        final_score = (success_rate * score_multiplier) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
        print("\n--- 🏆 FINAL SCORE ---")
        print(f"Calculation: ({success_rate:.2f} * {score_multiplier}) - ({total_wrong_guesses} * 5) - ({total_repeated_guesses} * 2)")
        print(f"Final Score: {final_score:.2f}")
    else:
        print("\n--- 🏆 FINAL SCORE (Formula mismatch) ---")
        print("Your test set was not 1000 or 2000 words.")
        print("Please report the stats above (Wins, Wrong Guesses) manually.")

if __name__ == "__main__":
    # --- We need to fix the DQN Agent to be on the right device ---
    # Open train_dqn.py and find the DQNAgent class.
    # Add this line to its __init__ method:
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # And change this line in its __init__ method:
    # self.qnetwork_local = DQN(state_size, action_size, seed).to(self.device)
    # self.qnetwork_target = DQN(state_size, action_size, seed).to(self.device)
    
    # --- AND in HangmanEnv class ---
    # Add this to its __init__ method:
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Let's just fix it in this script for now ---
    
    # Hotfix: Add device to env
    HangmanEnv.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hotfix: Add device to agent
    DQNAgent.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We must also move tensors to the device in the .act() method
    # The evaluation script handles this manually.
    
    evaluate_agent()