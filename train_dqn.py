import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from hangman_env import HangmanEnv  # Import our FINAL environment
import time

# --- 1. Set Parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BUFFER_SIZE = int(3e5)
BATCH_SIZE = 128
GAMMA = 0.99
LR = 5e-5               # Stay with the low LR
TAU = 1e-3
UPDATE_EVERY = 4

# Exploration: Longest Childhood
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
N_EPISODES = 20000      # Use all 20k episodes

# --- 2. Define the "Two-Tower" DQN (Unchanged, but now receives a larger stats_input) ---
class DQN_TwoTower(nn.Module):
    def __init__(self, stats_size, word_len, word_encoding_size, action_size=26, seed=42):
        super(DQN_TwoTower, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # --- Tower 1: Stats MLP ---
        # Input size is now 79 (1+26+26+26)
        self.stats_fc1 = nn.Linear(stats_size, 64)
        self.stats_fc2 = nn.Linear(64, 64)
        
        # --- Tower 2: Word 1D-CNN ---
        self.word_len = word_len
        self.word_encoding_size = word_encoding_size
        self.conv1 = nn.Conv1d(self.word_encoding_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.cnn_output_size = 128 * self.word_len
        
        # --- Final Head: Combined Network ---
        self.combined_fc1 = nn.Linear(64 + self.cnn_output_size, 256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, state):
        # --- Split the state vector ---
        # state is (batch_size, 727)
        
        # --- *** NEW SLICE *** ---
        # Stats Tower Input: (Batch, 79)
        stats_input = state[:, :79]
        
        # Word Tower Input: (Batch, 648)
        word_input = state[:, 79:].reshape(-1, self.word_len, self.word_encoding_size)
        # --- *** END NEW SLICE *** ---
        
        word_input = word_input.permute(0, 2, 1) 
        
        x_stats = F.relu(self.stats_fc1(stats_input))
        x_stats = F.relu(self.stats_fc2(x_stats))
        
        x_word = F.relu(self.conv1(word_input))
        x_word = F.relu(self.conv2(x_word))
        x_word = x_word.reshape(-1, self.cnn_output_size)
        
        x_combined = torch.cat((x_stats, x_word), dim=1)
        
        x = F.relu(self.combined_fc1(x_combined))
        x = F.relu(self.combined_fc2(x))
        return self.output(x)

# --- 3. Replay Buffer (Unchanged) ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=42):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)

# --- 4. Define the Agent (Unchanged logic, just new params) ---
class DQNAgent:
    def __init__(self, state_size, action_size, stats_size, word_len, word_encoding_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.stats_size = stats_size
        self.word_len = word_len
        self.word_encoding_size = word_encoding_size

        self.qnetwork_local = DQN_TwoTower(self.stats_size, self.word_len, self.word_encoding_size, action_size, seed).to(self.device)
        self.qnetwork_target = DQN_TwoTower(self.stats_size, self.word_len, self.word_encoding_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # --- *** NEW SLICE *** ---
        # state[1:27] is still the guessed_mask
        guessed_mask = state[1:27] 
        # --- *** END NEW SLICE *** ---
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        action_values[0][guessed_mask == 1] = -float('inf')
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            valid_actions = np.where(guessed_mask == 0)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            else:
                return 0

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# --- 5. The Training Loop ---
def train():
    # --- Get FINAL environment parameters ---
    env = HangmanEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    stats_size = env.stats_size
    word_len = env.max_word_len
    word_encoding_size = env.letter_encoding_size
    
    # --- Pass all params to the new Agent ---
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        stats_size=stats_size,
        word_len=word_len, 
        word_encoding_size=word_encoding_size
    )

    max_t = 32
    print_every = 100
    
    scores = []
    scores_window = deque(maxlen=100)
    wins_window = deque(maxlen=100)
    eps = EPS_START
    
    print(f"\n--- Starting Training (Final Agent) ---")
    print(f"Will train for {N_EPISODES} episodes.")
    start_time = time.time()

    for i_episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
                
        scores_window.append(score)
        scores.append(score)
        wins_window.append(1 if info.get('msg') == "Game won!" else 0)
        
        eps = max(EPS_END, EPS_DECAY * eps)
        
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window)
            win_rate = np.mean(wins_window) * 100
            elapsed = time.time() - start_time
            print(f"Ep {i_episode}\tAvg Score: {avg_score:.2f}\tWin Rate: {win_rate:.1f}%\tEps: {eps:.4f}\tTime: {elapsed:.0f}s")
            
        if i_episode % 2000 == 0:
            save_path = f"dqn_agent_final_ep{i_episode}.pth"
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            print(f"*** Model saved to {save_path} ***")

    print("\n--- Training Complete ---")
    torch.save(agent.qnetwork_local.state_dict(), "dqn_agent_final.pth")
    print("Final model (final_agent) saved to dqn_agent_final.pth")
    
    return scores

if __name__ == "__main__":
    scores = train()
    
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title('DQN Training Performance (Final Agent)')
        plt.savefig("training_plot.png")
        print("Training plot saved to training_plot.png")
    except ImportError:
        print("\n'matplotlib' not found. Skipping plot generation.")
