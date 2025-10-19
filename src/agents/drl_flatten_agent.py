import numpy as np
import random
from collections import namedtuple, deque
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim



BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size (reduced from 64 to save GPU memory)
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

# Use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache and set memory optimization if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"GPU Memory available: ~{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**2:.2f} MB")

class DRLFlattenAgent():
    """
    Deep Reinforcement Learning agent that processes grayscale (B/W) images for autonomous driving.
    
    This agent uses a bird's eye view grayscale camera image (84x84x1) with frame stacking (4 frames)
    to create a state representation of shape (84, 84, 4). The state is flattened to a 
    1D vector of 28,224 dimensions and processed through a fully-connected neural network.
    
    Architecture:
    - Input: Grayscale images (84x84x1) stacked over 4 frames → (84, 84, 4)
    - Flattened to: 28,224-dimensional vector (84 * 84 * 4)
    - Network: Fully-connected layers (fc1: 28224→64, fc2: 64→32, fc3: 32→action_size)
    - Learning: Deep Q-Network (DQN) with experience replay
    
    Converting to grayscale reduces dimensionality while retaining essential road structure
    information needed for navigation.
    """

    def __init__(self, env, seed):
        """Initialize an Agent object.
        
        Params
        ======
            env: The Gymnasium environment
            seed (int): random seed for reproducibility
        """
        # Get dimensions from environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Store the actual shape for later use in flattening
        self.state_shape = env.observation_space.shape
        
        # Calculate the flattened state size correctly
        state_size = int(np.prod(env.observation_space.shape))
        print(f"Calculated state_size: {state_size}")
        
        action_size = env.action_space.n
        
        # Store action_size for later use
        self.action_size = action_size
        
        self.seed = random.seed(seed)

        # Q-Network - Try GPU first, fallback to CPU if out of memory
        self.device = device  # Store device in instance
        try:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
            print(f"✓ Networks loaded on {self.device}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ GPU out of memory, falling back to CPU")
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
                self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
                self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
                print(f"✓ Networks loaded on CPU")
            else:
                raise e
                
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        
        # Save experience in replay memory
        state_copy = np.array(state).reshape(-1)
        next_state_copy = np.array(next_state).reshape(-1)
        
        self.memory.add(state_copy, action, reward, next_state_copy, done)

        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        # Flatten the state to match the network input shape
        state_array = np.array(state)
        # Check if we need to flatten the state
        if len(state_array.shape) > 1:
            state_flat = state_array.reshape(-1)
        else:
            state_flat = state_array
        
       
        state = torch.from_numpy(state_flat).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    


class QNetwork(nn.Module):
    """
    Q-Network for processing flattened grayscale image states.
    
    Architecture:
    - Input: Flattened state vector (84 * 84 * 4 = 28,224 dimensions)
    - Layer 1: Fully connected (28,224 → 64 units) + ReLU
    - Layer 2: Fully connected (64 → 32 units) + ReLU  
    - Output: Fully connected (32 → action_size)
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of flattened state (84*84*4 = 28,224 for grayscale)
            action_size (int): Dimension of each action
            seed (int): Random seed for reproducibility
            fc1_units (int): Number of nodes in first hidden layer (default: 64)
            fc2_units (int): Number of nodes in second hidden layer (default: 32)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Forward pass through the network.
        
        Maps flattened state (28,224 dimensions for grayscale) to Q-values for each action.
        
        Params
        ======
            state: Flattened grayscale image state tensor [batch_size, 28224]
            
        Returns
        =======
            action_values: Q-values for each action [batch_size, action_size]
        """
        x = F.relu(self.fc1(state))  # [batch_size, 64]
        x = F.relu(self.fc2(x))       # [batch_size, 32]
        return self.fc3(x)            # [batch_size, action_size]