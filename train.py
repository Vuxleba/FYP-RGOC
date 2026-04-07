import os
import argparse
import warnings
import random
from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from model import my_model, my_Q_net
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='facebook', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

# E net
parser.add_argument('--E_epochs', type=int, default=400, help='Number of epochs to train E.')
parser.add_argument('--n_input', type=int, default=1000, help='Number of units in hidden layer 1.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for hard cluster assignment in FCM.')

# Q net
parser.add_argument('--Q_epochs', type=int, default=40, help='Number of epochs to train Q.')
parser.add_argument('--epsilon', type=float, default=0.5, help='Greedy rate.')
parser.add_argument('--replay_buffer_size', type=float, default=100, help='Replay buffer size')
parser.add_argument('--Q_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--gamma', type=float, default=1.0, help='Resolution parameter for Fuzzy Modularity.')

args = parser.parse_args()

device = "cuda:0"
file_name = "result.csv"
ego_ids = ["107", "348", "414", "686", "698"]
datasets = [f"facebook_{ego_id}" for ego_id in ego_ids]

for args.dataset in datasets:
    file = open(file_name, "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset.startswith('facebook'):
        # Global defaults for all facebook datasets
        args.lr = 1e-3
        args.epsilon = 0.7
        args.replay_buffer_size = 100
        args.max_action = 10 

        ego_id = args.dataset.split('_')[1]
            
        if ego_id == '107':
            args.min_cluster = 5  
            args.n_input = -1   
            args.dims = [1500] 
            args.threshold = 0.95 
            args.gamma = 2.0
            args.gnnlayers = 2

        elif ego_id == '348':
            args.min_cluster = 10 
            args.n_input = 128    
            args.dims = [1280]
            args.threshold = 0.8  
            args.gnnlayers = 5   
            args.gamma = 1

        elif ego_id == '414':
            args.min_cluster = 3 
            args.n_input = -1     
            args.dims = [1500]
            args.threshold = 0.8
            args.gamma = 1.2
            args.gnnlayers = 3
            
        elif ego_id == '686':
            args.min_cluster = 10 
            args.n_input = -1     
            args.dims = [1500]
            args.threshold = 0.7
            args.gamma = 1.2
            args.gnnlayers = 1

        elif ego_id == '698':
            args.min_cluster = 9 
            args.n_input = -1     
            args.dims = [256]
            args.threshold = 0.9
            args.gamma = 1.8
            args.gnnlayers = 2

            
    nmi_list = []
    f1_list = []
    k_list = []
    # init
    for seed in range(10):
        setup_seed(seed)
        best_reward = float('-inf')
        X, y, A = load_graph_data(args.dataset, show_details=False) 
        features = X
        true_labels = y
        adj = sp.csr_matrix(A)
        
        if args.n_input != -1:
            pca = PCA(n_components=args.n_input)
            features = pca.fit_transform(features)

        A = torch.tensor(adj.todense()).float().to(device)

        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        print('Laplacian Smoothing...')
        adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        sm_fea_s = sp.csr_matrix(features).toarray()

        path = "dataset/facebook/{}_pca_{}_feat_sm_{}.npy".format(args.dataset, args.n_input, args.gnnlayers)
        if os.path.exists(path):
            sm_fea_s = np.load(path, allow_pickle=True)
        else:
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)
            np.save(path, sm_fea_s, allow_pickle=True)

        # X
        sm_fea_s = torch.FloatTensor(sm_fea_s) # shape = [n, d]
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

        args.cluster_num = np.random.randint(0, args.max_action) + args.min_cluster

        # init clustering
        _, _, u, predict_labels_matrix, _, _ = clustering(sm_fea_s.detach(), true_labels, args.cluster_num, device=device, threshold=args.threshold)
        best_nmi = 0
        best_f1 = 0
        best_cluster = 0

        # MLP
        if ego_id == '348' or ego_id == '414':
            model = my_model([sm_fea_s.shape[1]] + args.dims, act="ident")
        else:
            model = my_model([sm_fea_s.shape[1]] + args.dims)
        Q_net = my_Q_net(args.dims + [256, args.max_action]).to(device)
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

            weights = u.pow(2).T
            numerator = torch.mm(weights, state)
            denominator = weights.sum(dim=1, keepdim=True)
            cluster_state = numerator / (denominator + 1e-8)

            rand = False

            # do action by random choose
            if random.random() > tmp_epsilon:
                action = np.random.randint(0, args.max_action) 
                rand = True
            # do action by Q-net
            else:
                action = int(Q_net(state, cluster_state)[0:args.max_action].argmax())

            # Translate the 0-9 action into the actual cluster number
            args.cluster_num = action + args.min_cluster

            nmi, f1, u, predict_labels_matrix, centers, dis = clustering(state.detach(), true_labels, args.cluster_num, device=device, threshold=args.threshold)
            dis = (state.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1) + 1e-8 

            m = 2.0
            power = 2.0 / (m - 1)
            ratio = (dis.unsqueeze(2) / dis.unsqueeze(1)).pow(power)
            u = 1.0 / ratio.sum(dim=2)

            fcm_loss = (u.pow(m) * dis).sum(dim=1).mean()

            loss = infoNEC + 10 * fcm_loss 

            loss.backward()
            optimizer.step()

            # next state
            model.eval()
            z1, z2 = model(inx)
            next_state = (z1 + z2) / 2

            weights = u.pow(2).T
            numerator = torch.mm(weights, next_state)
            denominator = weights.sum(dim=1, keepdim=True)
            next_cluster_state = numerator / (denominator + 1e-8)

            k_degree = A_label.sum(dim=1, keepdim=True) # [N, 1]
            m2 = A_label.sum() + 1e-8 # 2 * total edges (scalar)
            actual_edges = torch.mm(u.t(), torch.mm(A_label, u))
            u_k = torch.mm(u.t(), k_degree) 
            expected_edges = torch.mm(u_k, u_k.t()) / m2 
            modularity_matrix = actual_edges - (args.gamma * expected_edges)
            fuzzy_modularity = torch.trace(modularity_matrix) / m2
            
            reward = 10 * fuzzy_modularity.detach()

            replay_buffer.append([[state.detach(), cluster_state.detach()], action,
                                  [next_state.detach(), next_cluster_state.detach()], reward])

            tmp_epsilon += epsilon_step

            if reward.item() > best_reward and rand == False:
                best_reward = reward.item()
                best_nmi = nmi
                best_f1 = f1
                best_cluster = args.cluster_num

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
                    loss_Q.backward()
                    optimizer_Q.step()
                    current_loss_Q = loss_Q.item()

                replay_buffer = []

        file = open(file_name, "a+")
        print(best_cluster, best_nmi, best_f1, file=file)
        file.close()
        nmi_list.append(best_nmi)
        f1_list.append(best_f1)
        k_list.append(best_cluster)

        tqdm.write("Optimization Finished!")
        tqdm.write('best_nmi: {}'.format(best_nmi))
        tqdm.write('best_f1: {}'.format(best_f1))

    nmi_list = np.array(nmi_list)
    f1_list = np.array(f1_list)
    k_list = np.array(k_list)

    file = open(file_name, "a+")
    print(args.gnnlayers, args.lr, file=file)
    print(k_list.mean(), k_list.std(), file=file)
    print(nmi_list.mean(), nmi_list.std(), file=file)
    print(f1_list.mean(), f1_list.std(), file=file)
    file.close()
