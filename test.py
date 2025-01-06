import pynvim
import string
import torch
import torch.nn as nn
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
nvim = pynvim.attach('socket', path='/tmp/nvim.sock')

def encode_text(line: str) -> List[int]:
    """convert text to list of int IDs"""
    ans = []
    for ch in line[:MAX_LEN]:
        ans.append(VOCAB_TO_ID.get(ch, 0))
    while len(ans) < MAX_LEN:
        ans.append(PAD_ID)
    return ans

def encode_mode(mode: str) -> int:
    if mode == 'n': return 0
    elif mode == 'i': return 1
    return 2

def state_to_tensor() -> torch.Tensor:
    text_line = nvim.current.buffer[0] # TODO adapt for :
    line_ids = encode_text(text_line)
    col = nvim.funcs.getpos('.')[2] # extract 3rd num
    cursor_encoded = [col / float(MAX_LEN)] # [2] # extract 3nd int
    mode_id = [encode_mode(nvim.funcs.mode())] 
    full_vec = line_ids + cursor_encoded + mode_id
    return torch.tensor(full_vec, dtype=torch.float32)

def step(action: str):
    """
    action is e.g. 'dw', 'x', 'iHello<Esc>', etc.
    """
    # 1. Apply action
    nvim.command('normal ' + action)
    
    # 2. Read new state
    updated_text = list(nvim.current.buffer[:])
    cursor_pos = nvim.funcs.getpos('.')
    mode = nvim.funcs.mode()
    
    # 3. Compute reward
    # For example, negative step penalty and check if we've reached some goal.
    reward, done = compute_reward(updated_text)
    
    # 4. Return (next_state, reward, done)
    return (updated_text, cursor_pos, mode), reward, done

def compute_reward(current_text):
    # TODO use diff instead and assign a % score
    if current_text == ["Hello, world!"]:
        return (100, True)
    else:
        return (-1, False)

torch.set_printoptions(precision=10)
print(state_to_tensor())


# # RL loop, simplified
# for episode in range(100):
#     done = False
#     state = (initial_text, (1,1), 'n')  # rough state
#     while not done:
#         # Agent picks an action
#         action = agent_policy(state)
#         state, reward, done = step(action)
#         agent_learn(state, action, reward)

