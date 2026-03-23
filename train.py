import os
import argparse
import warnings
from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_scatter import scatter
from model import my_model, my_Q_net
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

# E net
parser.add_argument('--E_epochs', type=int, default=400, help='Number of epochs to train E.')
parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')

# Q net
parser.add_argument('--Q_epochs', type=int, default=50, help='Number of epochs to train Q.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Greedy rate.')
parser.add_argument('--replay_buffer_size', type=float, default=50, help='Replay buffer size')
parser.add_argument('--Q_lr', type=float, default=1e-3, help='Initial learning rate.')


args = parser.parse_args()

device = "cuda:0"
file_name = "result.csv"
# for args.dataset in ["bat", "eat", "cora", "citeseer"]:
ego_ids = ["348", "414", "686", "698", "1684", "3980"]
datasets = [f"facebook_{ego_id}" for ego_id in ego_ids]

for args.dataset in datasets:
    # "amap",
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.5
        args.replay_buffer_size = 40

    elif args.dataset.startswith('facebook'):
        args.cluster_num = 12 # avg, will be adjusted dynamically anyway
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 0.85
        args.dims = [512]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = [500]
        args.epsilon = 0.7
        args.replay_buffer_size = 50

    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.3
        args.replay_buffer_size = 30

    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = [1500]
        args.epsilon = 0.7
        args.replay_buffer_size = 40

    nmi_list = []
    f1_list = []
    acc_list = []
    k_list = []
    # init
    for seed in range(10):
        setup_seed(seed)
        X, y, A = load_graph_data(args.dataset, show_details=False) 
        features = X
        true_labels = y
        adj = sp.csr_matrix(A)
        
        '''
        dont do pca for now
        if args.n_input != -1:
            pca = PCA(n_components=args.n_input)
            features = pca.fit_transform(features)
        '''
        

        A = torch.tensor(adj.todense()).float().to(device)

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        print('Laplacian Smoothing...')
        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()

        path = "dataset/facebook/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
        if os.path.exists(path):
            sm_fea_s = np.load(path, allow_pickle=True)
        else:
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)
            np.save(path, sm_fea_s, allow_pickle=True)

        # X
        sm_fea_s = torch.FloatTensor(sm_fea_s) # shape = [n, d]
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()


        args.cluster_num = np.random.randint(8, 19) + 2 #cluster range: 10-20

        # init clustering
        # _, _, predict_labels, _, _ = clustering(sm_fea_s.detach(), true_labels, args.cluster_num, device=device)
        _, _, _,  u, predict_labels_matrix, _, _ = clustering(sm_fea_s.detach(), true_labels, args.cluster_num, device=device)
        best_nmi = 0
        best_f1 = 0
        best_acc = 0
        best_epoch = 0
        best_cluster = 0
        best_reward = float('-inf') # Track the highest unsupervised reward
        # MLP
        if args.dataset == "citeseer":
            model = my_model([sm_fea_s.shape[1]] + args.dims, act="sigmoid")
        else:
            model = my_model([sm_fea_s.shape[1]] + args.dims)
        Q_net = my_Q_net(args.dims + [256, 19]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer_Q = optim.Adam(Q_net.parameters(), lr=args.Q_lr)

        model.to(device)
        inx = sm_fea_s.to(device)
        inx_origin = torch.FloatTensor(features).to(device)

        A_label = torch.FloatTensor(adj_1st).to(device)

        target = A_label
        mask = torch.ones([target.shape[0] * 2, target.shape[0] * 2]).to(device)
        mask -= torch.diag_embed(torch.diag(mask))

        tmp_epsilon = args.epsilon
        epsilon_step = (1 - tmp_epsilon) / args.E_epochs
        replay_buffer = []

        print('Start Training...')
        for epoch in tqdm(range(args.E_epochs)):
            model.train()
            Q_net.eval()
            optimizer.zero_grad()
            z1, z2 = model(inx) # z1.shape = [n, 512], z2.shape = [n, 1500] for facebook

            z1_z2 = torch.cat([z1, z2], dim=0)
            S = z1_z2 @ z1_z2.T

            # pos neg weight
            pos_neg = mask * torch.exp(S)

            pos = torch.cat([torch.diag(S, target.shape[0]), torch.diag(S, -target.shape[0])], dim=0)
            # pos weight
            pos = torch.exp(pos)

            neg = (torch.sum(pos_neg, dim=1) - pos)

            infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * target.shape[0])
            
            state = (z1 + z2) / 2

            """
            The change happens here
            """
            # cluster_state = scatter(state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean") #get the cluster centroids by averaging the node states in each cluster (based on nodes labels "predict_labels "), shape = [k, 512] for facebook
            '''
            pred_labels = torch.tensor(predict_labels_hard).long().to(device)

            one_hot = F.one_hot(pred_labels, num_classes=args.cluster_num).float()

            numerator = torch.mm(one_hot.T, state)
            denominator = one_hot.sum(dim=0, keepdim=True).T

            cluster_state = numerator / (denominator + 1e-8)
            '''
            
            # m=2 is standard for FCM and matches your clustering function
            weights = u.pow(2).T
            # Weighted sum: (Clusters, Samples) @ (Samples, Features) -> (Clusters, Features)
            numerator = torch.mm(weights, state)
            # Sum of weights per cluster: (Clusters, 1)
            denominator = weights.sum(dim=1, keepdim=True)
            # Normalized weighted average
            cluster_state = numerator / (denominator + 1e-8)

            rand = False

            # do action by random choose
            if random.random() > tmp_epsilon:
                action = np.random.randint(8, 19) # need fix later
                rand = True
            # do action by Q-net
            else:
                # Get argmax from range 8 to 18 (inclusive) for cluster_num adjustment
                action = int(Q_net(state, cluster_state)[8:19].argmax()) + 8

            args.cluster_num = action + 2
            nmi, f1, acc, u, predict_labels_matrix, centers, dis = clustering(state.detach(), true_labels, args.cluster_num, device=device)
            dis = (state.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1) + 1e-8 # why did they calculate dis again? Why not use the dis calculated in clustering function? Maybe they want to make sure the dis is calculated based on the current state and centers, not the one from clustering function which might be based on a different state/centers due to the way clustering is implemented.

            # q = dis / (dis.sum(-1).reshape(-1, 1))
            # p = q.pow(2) / q.sum(0).reshape(1, -1)
            # p = p / p.sum(-1).reshape(-1, 1)
            # pq_loss = F.kl_div(q.log(), p)
            # loss += 10 * pq_loss

            m = 2.0
            power = 2.0 / (m - 1)

            ratio = (dis.unsqueeze(2) / dis.unsqueeze(1)).pow(power)
            u = 1.0 / ratio.sum(dim=2)

            fcm_loss = (u.pow(m) * dis).sum(dim=1).mean() # later try 10 times here because currently the fcm_loss is not added to the total_loss somehow

            loss = infoNEC + 10 * fcm_loss 

            if reward.item() > best_reward and rand == False:
                best_reward = reward.item()
                best_nmi = nmi
                best_f1 = f1
                best_acc = acc
                best_cluster = args.cluster_num
                best_epoch = epoch + 1

            loss.backward()
            optimizer.step()

            # next state
            model.eval()
            z1, z2 = model(inx)
            next_state = (z1 + z2) / 2

            # next_cluster_state = scatter(next_state, torch.tensor(predict_labels).to(device), dim=0, reduce="mean")
            '''
            labels = torch.tensor(predict_labels_hard).long().to(device)
            one_hot = F.one_hot(labels, num_classes=args.cluster_num).float()

            numerator = torch.mm(one_hot.T, next_state)
            denominator = one_hot.sum(dim=0, keepdim=True).T

            next_cluster_state = numerator / (denominator + 1e-8)
            '''

            # m=2 is standard for FCM and matches your clustering function
            weights = u.pow(2).T
            # Weighted sum: (Clusters, Samples) @ (Samples, Features) -> (Clusters, Features)
            numerator = torch.mm(weights, next_state)
            # Sum of weights per cluster: (Clusters, 1)
            denominator = weights.sum(dim=1, keepdim=True)
            # Normalized weighted average
            next_cluster_state = numerator / (denominator + 1e-8)

            # center_dis = (centers.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1).mean()
            # reward = center_dis.detach() - torch.min(dis, dim=1).values.mean().detach()
            # fuzzy compactness
            # fuzzy_compactness = (u.pow(m) * dis).sum(dim=1).mean()
            # reward = center_dis.detach() - fuzzy_compactness.detach()
            '''
            Q_fuzzy reward
            '''
            # 1. Calculate node degrees and total edge weight from your adjacency matrix
            # A_label is shape [N, N]
            k_degree = A_label.sum(dim=1, keepdim=True) # [N, 1]
            m2 = A_label.sum() + 1e-8 # 2 * total edges (scalar)

            # 2. Term 1: Actual edges within fuzzy communities
            # torch.mm(A_label, u) -> [N, K]
            # u.t() @ [N, K] -> [K, K]
            actual_edges = torch.mm(u.t(), torch.mm(A_label, u))
            
            # 3. Term 2: Expected edges within fuzzy communities (Null Model)
            # u_k is the degree distribution per cluster -> [K, 1]
            u_k = torch.mm(u.t(), k_degree) 
            # Outer product of u_k divided by total edge weight -> [K, K]
            expected_edges = torch.mm(u_k, u_k.t()) / m2 

            # 4. Calculate Final Fuzzy Modularity (Q_F)
            # We take the trace (sum of the diagonal) to only reward edges 
            # that fall completely WITHIN the same cluster c.
            modularity_matrix = actual_edges - expected_edges
            fuzzy_modularity = torch.trace(modularity_matrix) / m2
            
            # The RL agent maximizes the reward. Modularity is higher when 
            # the community structure is strong.
            reward = 10 * fuzzy_modularity.detach()

            replay_buffer.append([[state.detach(), cluster_state.detach()], action,
                                  [next_state.detach(), next_cluster_state.detach()], reward])

            tmp_epsilon += epsilon_step

            # Placeholder for Q loss
            current_loss_Q = 0.0

            # replay_buffer full: train Q network
            if len(replay_buffer) >= args.replay_buffer_size:
                for it in range(args.Q_epochs):
                    model.eval()
                    Q_net.train()
                    optimizer_Q.zero_grad()
                    idx = list(range(args.replay_buffer_size))
                    np.random.shuffle(idx)
                    loss_Q = 0
                    for i in idx:
                        s = replay_buffer[i][0][0] # state
                        s_c = replay_buffer[i][0][1] # cluster state
                        a = replay_buffer[i][1] # action
                        s_new = replay_buffer[i][2][0] # next state
                        s_new_c = replay_buffer[i][2][1] # next cluster state
                        r = replay_buffer[i][3] # reward
                        Q_value = Q_net(s, s_c)[a] 
                        y = r + 0.1 * Q_net(s_new, s_new_c).max()
                        loss_Q += (y - Q_value) ** 2
                    loss_Q /= len(idx)
                    # loss_Q = loss_Q ** 0.5
                    # MSE loss
                    loss_Q.backward()
                    optimizer_Q.step()
                    current_loss_Q = loss_Q.item()

                    if (it + 1) % 10 == 0:
                        tqdm.write(f"    Q-Net Epoch {it + 1}/{args.Q_epochs}: Loss = {current_loss_Q:.4f}")
                # cleaning up
                replay_buffer = []

            # Print every 40 epochs
            if (epoch + 1) % 40 == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}: "
                    f"InfoNEC={infoNEC.item():.4f}, "
                    f"FCM Loss={fcm_loss.item():.4f}, "
                    f"Total Encoder Loss={loss.item():.4f}, "
                    f"Reward={reward.item():.4f}, "
                )

        file = open(file_name, "a+")
        print(best_cluster, best_nmi, best_f1, best_acc, best_epoch, best_reward, file=file)
        file.close()
        nmi_list.append(best_nmi)
        f1_list.append(best_f1)
        acc_list.append(best_acc)
        k_list.append(best_cluster)

        tqdm.write("Optimization Finished!")
        tqdm.write('best_nmi: {}'.format(best_nmi))
        tqdm.write('best_f1: {}'.format(best_f1))
        tqdm.write('best_acc: {}'.format(best_acc))
        tqdm.write('best_epoch: {}'.format(best_epoch))

    nmi_list = np.array(nmi_list)
    f1_list = np.array(f1_list)
    acc_list = np.array(acc_list)
    k_list = np.array(k_list)

    file = open(file_name, "a+")
    print(args.gnnlayers, args.lr, file=file)
    print(k_list.mean(), k_list.std(), file=file)
    print(nmi_list.mean(), nmi_list.std(), file=file)
    print(f1_list.mean(), f1_list.std(), file=file)
    print(acc_list.mean(), acc_list.std(), file=file)
    file.close()
