import numpy as np
from hmmlearn import hmm
from collections import defaultdict
import pickle
import re
import time

# --- Paste the function from Step 1 here ---
def load_and_group_corpus(filepath="corpus.txt"):
    words_by_length = defaultdict(list)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip().upper()
                if word.isalpha():
                    words_by_length[len(word)].append(word)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None
    print(f"Corpus loaded. Found words for {len(words_by_length)} different lengths.")
    return words_by_length

# --- Helper function to convert words to number sequences ---
def words_to_sequences(word_list):
    """
    Converts a list of words ['APPLE', 'APPLY'] into a format
    hmmlearn can understand:
    - A single concatenated array of observations: [0, 15, 15, 11, 4, 0, 15, 15, 11, 24]
    - A list of lengths: [5, 5]
    """
    sequences = []
    lengths = []
    for word in word_list:
        # Convert 'A'->0, 'B'->1, ..., 'Z'->25
        seq = [ord(char) - ord('A') for char in word]
        sequences.extend(seq)
        lengths.append(len(word))
        
    # Reshape into a column vector as required by hmmlearn
    return np.array(sequences).reshape(-1, 1), lengths

# --- Main Training Function ---
def train_all_hmms(grouped_words, n_hidden_states=20, n_iter=100):
    """
    Trains one HMM for each word length and returns a dictionary of models.
    """
    hmm_oracles = {}
    
    # Iterate over each length (e.g., 5: ['APPLE', ...])
    for length, words in grouped_words.items():
        if not words:
            continue
            
        print(f"Training HMM for length {length} (on {len(words)} words)...")
        start_time = time.time()
        
        # 1. Format the data for hmmlearn
        # This converts ['APPLE', 'APPLY'] into (observations, lengths)
        sequences, lengths = words_to_sequences(words)
        
        # 2. Initialize the HMM
        # We use CategoricalHMM, which is correct for discrete symbols (0-25)
        # n_features = 26 (A-Z)
        model = hmm.CategoricalHMM(
            n_components=n_hidden_states,
            n_features=26,  # <-- This is the critical fix!
            n_iter=n_iter,
            tol=1e-3,
            verbose=False,
            random_state=42
        )
        
        # 3. Train the model
        # The .fit() method runs the Baum-Welch algorithm
        try:
            model.fit(sequences, lengths)
            hmm_oracles[length] = model
            end_time = time.time()
            print(f"  -> Done in {end_time - start_time:.2f}s. Model converged: {model.monitor_.converged}")
        except Exception as e:
            print(f"  -> ERROR training model for length {length}: {e}")
            # This can happen if a length has too few samples (e.t., only one 25-letter word)
            # We simply skip saving a model for this length.
            
    return hmm_oracles

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    grouped_words = load_and_group_corpus("corpus.txt")
    
    if grouped_words:
        # 2. Train models
        
        # --- *** THE CHANGES ARE HERE *** ---
        # We are creating a "smarter" oracle with more states and more training time.
        N_STATES = 30  # <-- WAS 20
        N_ITER = 200   # <-- WAS 100
        # --- *** END CHANGES *** ---
        
        print(f"\nStarting HMM training with {N_STATES} hidden states and {N_ITER} iterations...")
        
        # Pass both parameters to the function
        all_models = train_all_hmms(grouped_words, n_hidden_states=N_STATES, n_iter=N_ITER)
        
        # 3. Save all trained models to a single file
        if all_models:
            output_file = "hmm_oracles.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(all_models, f)
            print(f"\nSuccessfully trained and saved {len(all_models)} HMM models to '{output_file}'.")
        else:
            print("\nNo models were trained.")
