import os
from collections import defaultdict
import torch
import random
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def load_graph_data():
    pass

def load_facebook_data():
    pass

def normalize_adj():
    pass

def diffusion_adj():
    pass