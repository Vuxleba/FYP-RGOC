import torch
import torch.nn as nn
import random

# ==========================================
# 0. SETUP DUMMY Q-NETWORK & VARIABLES
# ==========================================
class MockQNet(nn.Module):
    def __init__(self, hidden_dim=32, num_actions=9):
        super().__init__()
        # LazyLinear lets us infer the input dimension from the first batch.
        self.state_encoder = nn.LazyLinear(hidden_dim)
        self.cluster_encoder = nn.LazyLinear(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, s, s_c):
        # Encode node and cluster representations separately, then merge summaries.
        state_feat = torch.relu(self.state_encoder(s))
        cluster_feat = torch.relu(self.cluster_encoder(s_c))
        joint_summary = torch.relu(
            state_feat.mean(dim=0, keepdim=True) + cluster_feat.mean(dim=0, keepdim=True)
        )
        q_values = self.value_head(joint_summary)
        return q_values.repeat(s.shape[0], 1)

Q_net = MockQNet()
optimizer_Q = torch.optim.SGD(Q_net.parameters(), lr=0.01)
replay_buffer = []
epsilon = 0.5

print("Starting MDP Cycle...")
print("-" * 40)

# ==========================================
# [MDP STEP 1: CURRENT STATE (S_t)]
# ==========================================
# Imagine we have 5 data points, each with 4 features.
# Currently, they are grouped into 3 clusters.
state = torch.rand(5, 4) 
cluster_state = torch.rand(3, 4) 

print("[STEP 1: STATE]")
print(f"-> 'state' tensor shape (5 points, 4 features): {list(state.shape)}")
print(f"-> 'cluster_state' tensor shape (3 clusters, 4 features): {list(cluster_state.shape)}\n")

# ==========================================
# [MDP STEP 2: ACTION (A_t)]
# ==========================================
# The agent decides how many clusters to use next (Action space: 0-8 -> 2-10 clusters)
if random.random() > epsilon:
    action = random.randint(0, 8)
    print("[STEP 2: ACTION] -> Randomly chosen (Epsilon-Greedy)")
else:
    # Q_net returns [5, 9]. .mean(0) averages across the batch to get [9] Q-values.
    action = int(Q_net(state, cluster_state).mean(0).argmax())
    print("[STEP 2: ACTION] -> Chosen by Q-Network")

cluster_num = action + 2
print(f"-> Action Index: {action}")
print(f"-> New Target Cluster Count: {cluster_num}\n")

# ==========================================
# [MDP STEP 3: REWARD & NEXT STATE (R_t & S_{t+1})]
# ==========================================
# In reality, the environment runs clustering here. We just mock the results.
reward = torch.tensor(1.25) # Mock reward based on cluster separation

next_state = torch.rand(5, 4) # Data points get new embeddings
# CRITICAL: The next cluster state size depends entirely on the action taken!
next_cluster_state = torch.rand(cluster_num, 4) 

print("[STEP 3: REWARD & NEXT STATE]")
print(f"-> Reward scalar: {reward.item()}")
print(f"-> 'next_state' shape remains: {list(next_state.shape)}")
print(f"-> 'next_cluster_state' shape dynamically changed to match action: {list(next_cluster_state.shape)}\n")

# ==========================================
# [MDP STEP 4: EXPERIENCE REPLAY & LEARNING]
# ==========================================
# Store the transition
replay_buffer.append([
    (state, cluster_state), 
    action, 
    (next_state, next_cluster_state), 
    reward
])

print("[STEP 4: Q-LEARNING UPDATE]")
print(f"-> Replay Buffer Size: {len(replay_buffer)}")

# Simulate training loop (assuming buffer is full at 1 for this demo)
optimizer_Q.zero_grad()

# Unpack a memory
(s, s_c), a, (s_new, s_new_c), r = replay_buffer[0]

# Bellman Equation: Q(s,a) = r + gamma * max(Q(s', a'))
current_Q_value = Q_net(s, s_c).mean(0)[a]
target_Q_value = r + 0.1 * Q_net(s_new, s_new_c).mean(0).max()

loss_Q = (target_Q_value - current_Q_value) ** 2

print(f"-> Current Q-Value for action {a}: {current_Q_value.item():.4f}")
print(f"-> Target Q-Value (Reward + Discounted Future): {target_Q_value.item():.4f}")
print(f"-> Q-Network Loss (MSE): {loss_Q.item():.4f}")

loss_Q.backward()
optimizer_Q.step()
print("-> Q-Network weights updated successfully.")