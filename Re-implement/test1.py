import torch
import torch.nn as nn
import random

# init clustering state
predict_labels = torch.tensor([0, 0, 1, 1, 2])  # 5 data points in 3 clusters

# calculate cluster_state by using scatter. What scatter does is grouping the node embeddings by the assigned clusters 
# and calculates the centroid (mean in this case) for each cluster. From N x d to K x d.
# cluster_state = scatter(state, predict_labels.to_device(), dim=0, reduce='mean')

# agent choose action: 1. random or 2. adopt Q_net's action:
# action = int(Q_net(state, cluster_state).mean(0).argmax())

# calculate next stage, and next cluster_stage set model.eval() (after model's loss.backward())

#calculate reward
# reward = center_dis.detach() - torch.min(dis, dim=1).values.mean().detach()
# this is maximizing distance between clusters head and minimizing nodes inside a cluster

# add replay_buffer.append([[state.detach(), cluster_state.detach()], action,[next_state.detach(), next_cluster_state.detach()], reward])

# train Q_net when replay_buffer is full (after replay_buffer_size times)