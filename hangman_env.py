import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import string
import random
from collections import defaultdict, Counter
from scipy.special import logsumexp
import torch

# --- NEW: Load corpus, max_len, AND frequencies ---
def _load_corpus_and_frequencies(filepath="corpus.txt"):
    words = []
    max_len = 0
    letter_counts = Counter()
    total_letters = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip().upper()
                if word.isalpha():
                    words.append(word)
                    if len(word) > max_len:
                        max_len = len(word)
                    for char in word:
                        letter_counts[char] += 1
                        total_letters += 1
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, 0, None
        
    # Create the frequency probability vector (A-Z)
    freq_vector = np.zeros(26)
    for i, char in enumerate(string.ascii_uppercase):
        freq_vector[i] = letter_counts[char] / total_letters
        
    return words, max_len, freq_vector

# --- Manual Forward-Backward Implementation (Unchanged) ---
def _log_forward_pass(log_framelogprob, log_startprob, log_transmat):
    n_observations, n_components = log_framelogprob.shape
    log_alpha = np.zeros((n_observations, n_components))
    log_alpha[0, :] = log_startprob + log_framelogprob[0, :]
    for t in range(1, n_observations):
        for j in range(n_components):
            log_alpha[t, j] = log_framelogprob[t, j] + logsumexp(log_transmat[:, j] + log_alpha[t-1, :])
    return log_alpha

def _log_backward_pass(log_framelogprob, log_startprob, log_transmat):
    n_observations, n_components = log_framelogprob.shape
    log_beta = np.zeros((n_observations, n_components))
    for t in range(n_observations - 2, -1, -1):
        for i in range(n_components):
            log_beta[t, i] = logsumexp(log_transmat[i, :] + log_framelogprob[t+1, :] + log_beta[t+1, :])
    return log_beta

class HangmanEnv(gym.Env):
    
    def __init__(self, oracle_path="hmm_oracles.pkl", corpus_path="corpus.txt"):
        super(HangmanEnv, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading HMM Oracles...")
        try:
            with open(oracle_path, 'rb') as f:
                self.hmm_oracles = pickle.load(f)
        except FileNotFoundError:
            raise
        print(f"Successfully loaded {len(self.hmm_oracles)} HMM models.")
        
        # --- NEW: Load all corpus data ---
        self.word_list, self.max_word_len, self.freq_vector = _load_corpus_and_frequencies(corpus_path)
        if not self.word_list:
            raise ValueError("Corpus list is empty.")
        print(f"Loaded {len(self.word_list)} words. Max length: {self.max_word_len}. Frequencies calculated.")

        self.max_lives = 6
        self.alphabet = string.ascii_uppercase
        self.n_letters = len(self.alphabet) # 26
        self.letter_encoding_size = self.n_letters + 1 # 27 (A-Z + BLANK)
        
        # --- NEW: Final State Space Definition ---
        self.action_space = spaces.Discrete(self.n_letters)

        # Observation (State) Space:
        # 1. Lives left (normalized)   : 1 element
        # 2. Guessed mask (binary)     : 26 elements
        # 3. HMM probabilities (float) : 26 elements
        # 4. Freq probabilities (float): 26 elements  <--- NEW
        # 5. Masked Word (one-hot)     : max_word_len * 27 elements
        
        self.stats_size = 1 + self.n_letters + self.n_letters + self.n_letters # 1 + 26 + 26 + 26 = 79
        self.word_size = self.max_word_len * self.letter_encoding_size # e.g., 24 * 27 = 648
        self.state_size = self.stats_size + self.word_size # 79 + 648 = 727
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_size,), dtype=np.float32
        )
        print(f"Final state vector size: {self.state_size} (Stats: {self.stats_size}, Word: {self.word_size})")

        # Init game state
        self.secret_word = ""
        self.masked_word = ""
        self.lives_left = 0
        self.guessed_letters = set()
        
        # Init HMM state
        self.log_startprob = None
        self.log_transmat = None
        self.log_emissionprob = None
        self.n_components = 0

    def _get_hmm_probs(self):
        # (This function is unchanged)
        if self.log_startprob is None:
            return np.ones(self.n_letters) / self.n_letters 

        word_len = len(self.secret_word)
        log_framelogprob = np.zeros((word_len, self.n_components))
        wrong_guesses = self.guessed_letters - set(self.secret_word)
        available_mask = np.ones(self.n_letters, dtype=bool)
        for letter in wrong_guesses:
            available_mask[ord(letter) - ord('A')] = False

        for t in range(word_len):
            char = self.masked_word[t]
            if char != '_':
                obs_index = ord(char) - ord('A')
                log_framelogprob[t, :] = self.log_emissionprob[:, obs_index]
            else:
                log_framelogprob[t, :] = logsumexp(self.log_emissionprob[:, available_mask], axis=1)

        try:
            log_fwd = _log_forward_pass(log_framelogprob, self.log_startprob, self.log_transmat)
            log_bwd = _log_backward_pass(log_framelogprob, self.log_startprob, self.log_transmat)
        except Exception as e:
            return np.ones(self.n_letters) / self.n_letters 

        log_gamma = log_fwd + log_bwd
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        final_prob_vector = np.zeros(self.n_letters)
        for t in range(word_len):
            if self.masked_word[t] == '_':
                letter_probs_at_t = gamma[t, :] @ np.exp(self.log_emissionprob)
                final_prob_vector += letter_probs_at_t

        for letter in self.guessed_letters:
            final_prob_vector[ord(letter) - ord('A')] = 0
            
        final_prob_vector[final_prob_vector < 0] = 0
        total_prob = final_prob_vector.sum()
        if total_prob > 0:
            final_prob_vector = final_prob_vector / total_prob
        else:
            for i in range(self.n_letters):
                if i not in [ord(l) - ord('A') for l in self.guessed_letters]:
                    final_prob_vector[i] = 1
            total_prob = final_prob_vector.sum()
            if total_prob > 0:
                final_prob_vector = final_prob_vector / total_prob
            else:
                final_prob_vector[:] = 0

        return final_prob_vector

    def _get_state(self):
        """
        Constructs the FINAL state vector.
        """
        # 1. Lives (normalized)
        lives_vec = np.array([self.lives_left / self.max_lives], dtype=np.float32)
        
        # 2. Guessed mask (binary)
        guessed_vec = np.zeros(self.n_letters, dtype=np.float32)
        for letter in self.guessed_letters:
            guessed_vec[ord(letter) - ord('A')] = 1.0
            
        # 3. HMM probabilities
        hmm_probs = self._get_hmm_probs().astype(np.float32)
        
        # 4. --- NEW: Frequency probabilities ---
        # We copy the base frequency vector
        freq_probs = np.copy(self.freq_vector).astype(np.float32)
        # We mask out guessed letters so the agent doesn't have to learn this
        freq_probs[guessed_vec == 1] = 0
        # Re-normalize
        freq_sum = freq_probs.sum()
        if freq_sum > 0:
            freq_probs = freq_probs / freq_sum
        
        # 5. --- NEW: One-hot encoded masked word ---
        word_encoding = np.zeros((self.max_word_len, self.letter_encoding_size), dtype=np.float32)
        for i, char in enumerate(self.masked_word):
            if char == '_':
                word_encoding[i, 26] = 1.0 # Index 26 is for BLANK
            else:
                word_encoding[i, ord(char) - ord('A')] = 1.0
        for i in range(len(self.masked_word), self.max_word_len):
            word_encoding[i, 26] = 1.0 # Pad with BLANK
        word_vec = word_encoding.flatten()
        
        # Concatenate all parts (Stats Tower first, then Word Tower)
        state = np.concatenate([lives_vec, guessed_vec, hmm_probs, freq_probs, word_vec])
        return state

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.secret_word = random.choice(self.word_list)
        word_len = len(self.secret_word)
        
        while word_len not in self.hmm_oracles:
            self.secret_word = random.choice(self.word_list)
            word_len = len(self.secret_word)
            
        model = self.hmm_oracles[word_len]
        
        log_floor = -1e10 
        with np.errstate(divide='ignore'):
            self.log_startprob = np.log(model.startprob_)
            self.log_transmat = np.log(model.transmat_)
            self.log_emissionprob = np.log(model.emissionprob_)

        self.log_startprob[self.log_startprob == -np.inf] = log_floor
        self.log_transmat[self.log_transmat == -np.inf] = log_floor
        self.log_emissionprob[self.log_emissionprob == -np.inf] = log_floor
        self.n_components = model.n_components
        
        self.masked_word = "_" * word_len
        self.lives_left = self.max_lives
        self.guessed_letters = set()
        
        initial_state = self._get_state()
        info = {"masked_word": self.masked_word, "secret_word": self.secret_word}
        
        return initial_state, info

    def step(self, action):
        # (This function is unchanged, with the +15 / -5 rewards)
        letter = self.alphabet[action]
        
        if letter in self.guessed_letters:
            reward = -50 
            done = False
            new_state = self._get_state()
            info = {"msg": "Repeated guess"}
            return new_state, reward, done, False, info

        self.guessed_letters.add(letter)

        if letter in self.secret_word:
            reward = 15  # <-- Correct +15 reward
            new_masked_word = list(self.masked_word)
            for i, char in enumerate(self.secret_word):
                if char == letter:
                    new_masked_word[i] = letter
            self.masked_word = "".join(new_masked_word)
            
            if "_" not in self.masked_word:
                reward = 100
                done = True
                info = {"msg": "Game won!"}
            else:
                done = False
                info = {"msg": "Correct guess"}
        
        else:
            self.lives_left -= 1
            reward = -5 # <-- Correct -5 reward
            
            if self.lives_left <= 0:
                reward = -100
                done = True
                info = {"msg": "Game lost!"}
            else:
                done = False
                info = {"msg": "Wrong guess"}

        new_state = self._get_state()
        
        return new_state, reward, done, False, info

# --- Test Block (Updated) ---
if __name__ == "__main__":
    
    print("Testing HangmanEnv with FINAL 'Hybrid Intuition' State...")
    env = HangmanEnv(oracle_path="hmm_oracles.pkl", corpus_path="corpus.txt")
    
    state, info = env.reset()
    
    print(f"Game started. Secret word: {info['secret_word']}")
    print(f"Mask: {info['masked_word']}")
    print(f"Final state vector shape: {state.shape}")
    print(f"Calculated state size: {env.state_size}")
    assert state.shape[0] == env.state_size, "State size mismatch!"
    
    action = env.action_space.sample()
    print(f"\nTaking random action: Guessing '{env.alphabet[action]}'")
    
    new_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"New mask: {env.masked_word}")
    print(f"Reward: {reward}")
    print(f"Game over: {terminated}")
    print(f"New state vector shape: {new_state.shape}")
    
    print("\nEnvironment test complete.")
