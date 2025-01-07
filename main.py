import pynvim
import string
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from typing import List

ACTIONS = [
    "h",
    "j",
    "k",
    "l",
    "x",
    "i",
]

VOCAB = list(string.printable)
VOCAB_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
PAD_ID = len(VOCAB)
MAX_LEN = 40

# create a headless nvim instance
# nvim = pynvim.attach('child', argv=["nvim", "--headless", "--embed"])

# attach to an existing socket
nvim = pynvim.attach("socket", path="/tmp/nvim.sock")


def encode_text(line: str) -> List[int]:
    """convert text to list of int IDs"""
    ans = []
    for ch in line[:MAX_LEN]:
        ans.append(VOCAB_TO_ID.get(ch, 0))
    while len(ans) < MAX_LEN:
        ans.append(PAD_ID)
    return ans


def encode_mode(mode: str) -> int:
    if mode == "n":
        return 0
    elif mode == "i":
        return 1
    return 2


def state_to_tensor() -> torch.Tensor:
    text_line = nvim.current.buffer[0]  # TODO adapt for :
    line_ids = encode_text(text_line)
    col = nvim.funcs.getpos(".")[2]  # extract 3rd num
    cursor_encoded = [col / float(MAX_LEN)]  # [2] # extract 3nd int
    mode_id = [encode_mode(nvim.funcs.mode())]
    full_vec = line_ids + cursor_encoded + mode_id
    return torch.tensor(full_vec, dtype=torch.float32)


def step(action: str):
    """
    action is e.g. 'dw', 'x', 'iHello<Esc>', etc.
    """
    # 1. Apply action
    nvim.command("normal " + action)

    # 2. Read new state
    updated_text = list(nvim.current.buffer[:])
    cursor_pos = nvim.funcs.getpos(".")
    mode = nvim.funcs.mode()

    # 3. Compute reward
    # For example, negative step penalty and check if we've reached some goal.
    reward, done = compute_reward(updated_text)

    # 4. Return (reward, done)
    return reward, done


def compute_reward(current_text: List[str]):
    # TODO use diff instead and assign a % score
    if current_text == ["hjk"]: # TODO make this work with arbitrary goal
        return (100, True)
    else:
        return (-1, False)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        """
        TODO what does this mean
        - state_dim = MAX_LEN + 1 (cursor) + 1 (mode) = 40 + 1 + 1 = 42
        - action_dim = len(ACTIONS) = 6
        x shape: (batch_size, state_dim)
        returns: (batch_size, action_dim)  # raw logits
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

state_dim = MAX_LEN + 1 + 1
action_dim = len(ACTIONS)
policy_net = PolicyNet(state_dim, action_dim)

# This chooses a random action?
def agent_policy():
    state_tensor = state_to_tensor().unsqueeze(0)
    logits = policy_net(state_tensor)
    action_dist = dist.Categorical(logits=logits)
    action_idx = int(action_dist.sample().item()) # take a sample
    return ACTIONS[action_idx]

torch.set_printoptions(precision=10)
print(state_to_tensor())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99

def run_episode(max_steps=50):
    """
    Runs one episode in the environment.
    Collect (state, action, reward) pairs until done or max_steps.
    Return the trajectory.
    """
    log_probs = []
    rewards = []
    done = False
    for t in range(max_steps):
        state_tensor = state_to_tensor().unsqueeze(0)
        logits = policy_net(state_tensor)
        action_dist = dist.Categorical(logits=logits)

        action_idx = action_dist.sample()
        action = ACTIONS[int(action_idx.item())]
        log_prob = action_dist.log_prob(action_idx)
        reward, done = step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break
    return log_probs, rewards

def compute_returns(rewards, gamma=0.99):
    ans = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        ans.insert(0, R)
    return ans

def main():
    for episode in range(1000):
        log_probs, rewards = run_episode()  # you'll need a real env
        returns = compute_returns(rewards, gamma)
        
        # Convert returns to a torch tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns (common trick for stable training)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Calculate the policy loss
        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update the network
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Possibly print diagnostics
        if episode % 50 == 0:
            print(f"Episode {episode}, total reward: {sum(rewards)}")

if __name__ == "__main__":
    main()

# # RL loop, simplified
# for episode in range(100):
#     done = False
#     state = (initial_text, (1,1), 'n')  # rough state
#     while not done:
#         # Agent picks an action
#         action = agent_policy(state)
#         state, reward, done = step(action)
#         agent_learn(state, action, reward)
