"""

Created on May 17, 2022
Implementation of TopicVAE in
Zhiqiang Guo et al. MDGCF: Multi-Dependency Graph Collaborative Filtering with Neighborhood- and Homogeneous-level Dependencies

@author: Zhiqiang Guo (zhiqiangguo@hust.edu.cn)

"""

import os
import sys
import random
import argparse
import cppimport
import numpy as np
from time import time
import scipy.sparse as sp
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd() + '/sources')
sampling = cppimport.imp("sampling")

parser = argparse.ArgumentParser(description="MDGCF")
parser.add_argument('--task', type=str, default='train', help='the current task: train or test.')
parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
parser.add_argument('--gpu_id', type=int, default=0, help="if -1 use cpu.")
parser.add_argument('--dataset', type=str, default='AMusic', help="datasets: [AMusic, AKindle, Gowalla].")
parser.add_argument('--topks', nargs='?', default="[10, 20]", help="@k test list.")
parser.add_argument('--epochs', type=int, default=1000, help='the number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=2048, help="the batch size for bpr loss training procedure.")
parser.add_argument('--initilizer', type=str, default='normal', help='the initilizer for embedding, support [xavier, normal, pretrain].')
parser.add_argument('--emb_size', type=int, default=64, help="the embedding size.")
parser.add_argument('--layer', type=int, default=4, help="the layer num.")
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate.")
parser.add_argument('--decay', type=float, default=0.0001, help="the weight decay for L2 normalizaton.")
parser.add_argument('--early_stop', type=int, default=50, help='early_stop')
parser.add_argument('--top_H', type=int, default=4, help='the top number of similar user or item.')
parser.add_argument('--alpha', type=float, default=0.2, help='the alpha parameter.')
parser.add_argument('--beta', type=float, default=0.3, help='the beta parameter.')
parser.add_argument('--self_loop', type=int, default=0, help='if consider the node itself to the similarity matrix.')
args = parser.parse_args()
print(args)
topks = eval(args.topks)

random.seed(args.seed)
sampling.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

str_dev = "cuda:" + str(args.gpu_id)
device = torch.device(str_dev if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu")
print('Device:', device)

root_path = "./"
data_path = root_path + 'data/'
model_path = root_path + 'checkpoints/'
log_path = root_path + 'logs/'
# emb_path = root_path + 'embs/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
# if not os.path.exists(emb_path):
#     os.makedirs(emb_path)

# modelfile
model_file = model_path + args.dataset + '_' + str(args.emb_size) + '_' \
             + str(args.layer) + '.pth.tar'
# logfile
log_file = "_es%d_l%d_lr%.3f_re%.4f_al%.1f_be%.1f_H%d_sl%d_sd%d_%s.txt" \
           % (args.emb_size, args.layer, args.lr, args.decay, args.alpha,
              args.beta, args.top_H, args.self_loop, args.seed, datetime.now().strftime("%y%m%d%H%M%S"))
_log = open(log_path + args.dataset + log_file, 'w')
_log.write(str(args) + '\n')

def load():
    path = "./data/" + args.dataset
    train_file = path + '/train.txt'
    test_file = path + '/test.txt'
    group_file = path + '/group.txt'
    print(f'loading [{path}]')
    n_user = 0
    m_item = 0
    num_traindata = 0
    trainUser, trainItem = [], []
    trainData = []
    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                user = int(l[0])
                trainUser.extend([user] * len(items))
                trainItem.extend(items)
                trainData.append(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, user)
                num_traindata += len(items)
    num_testdata = 0
    testData = []
    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                user = int(l[0])
                testData.append(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, user)
                num_testdata += len(items)
    m_item += 1
    n_user += 1
    print(f"user:{n_user}, item:{m_item}, "
          f"interaction: {num_traindata + num_testdata}, "
          f"Sparsity : {(num_traindata + num_testdata) / n_user / m_item}")
    print(f"{num_traindata} interactions for training")
    print(f"{num_testdata} interactions for testing")

    with open(group_file, 'r') as f:
        group = eval(f.readline())

    # (users,items), bipartite graph
    graph = sp.csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(n_user, m_item))
    graph = graph.todok()
    # graph = graph + sp.eye(graph.shapes[0])
    d_user = np.array(graph.sum(axis=1))
    d_user = np.power(d_user, -0.5).flatten()
    d_user[np.isinf(d_user)] = 0.
    d_user = sp.diags(d_user)
    d_item = np.array(graph.sum(axis=0))
    d_item = np.power(d_item, -0.5).flatten()
    d_item[np.isinf(d_item)] = 0.
    d_item = sp.diags(d_item)
    graph = d_user.dot(graph)
    graph = graph.dot(d_item)
    graph = graph.tocsr()
    graph = _convert_sp_mat_to_sp_tensor(graph)
    adj = graph.coalesce().to(device)
    return n_user, m_item, num_traindata, trainData, testData, adj, group

def _convert_sp_mat_to_sp_tensor(x):
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def sp_func(x):
    return torch.log(1+torch.exp(x))

class MDGCF(nn.Module):
    def __init__(self, num_users, num_items, adj):
        super(MDGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.flag = 0
        self.__init_weight()
        self.adj = adj
        self.act = nn.Sigmoid()
        self.sim_adj = None
        self.user_sim = None
        self.item_sim = None
        self.user_sim_adj = None
        self.item_sim_adj = None

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.emb_size)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=args.emb_size)
        if args.initilizer == 'xavier':
            nn.init.xavier_uniform_(self.embedding_user.weight)
            nn.init.xavier_uniform_(self.embedding_item.weight)
            print('use xavier initilizer')
        elif args.initilizer == 'normal':
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use normal initilizer')
        elif args.initilizer == 'pretrain':
            emb_user = np.load('data/' + args.dataset + '/embedding_user.npy')
            emb_item = np.load('data/' + args.dataset + '/embedding_item.npy')
            self.embedding_user.weight.data.copy_(torch.from_numpy(emb_user))
            self.embedding_item.weight.data.copy_(torch.from_numpy(emb_item))
            print('use pretarined embedding')

    def cosine_similarity(self, x, y):
        """
            get the cosine similarity between to matrix
            consin(x, y) = xy / (sqrt(x^2) * sqrt(y^2))
        """
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        xy = torch.matmul(x, y.transpose(0, 1))
        x_norm = torch.sqrt(torch.mul(x, x).sum(1))
        y_norm = torch.sqrt(torch.mul(y, y).sum(1))
        x_norm = 1.0 / (x_norm.unsqueeze(1) + 1e-8)
        y_norm = 1.0 / (y_norm.unsqueeze(0) + 1e-8)
        # xy = torch.mul(torch.mul(xy, x_norm), y_norm)
        l = 5
        num_b = x.shape[0] // l
        if num_b * l < x.shape[0]:
            l = l + 1
        for i in range(l):
            begin = i * num_b
            end = (i + 1) * num_b
            end = xy.shape[0] if end > xy.shape[0] else end
            xy[begin:end] = torch.mul(torch.mul(xy[begin:end], x_norm[begin:end]), y_norm)
        return xy

    def update_flag(self, epoch):
        if args.alpha == 0.0 and epoch != 0:
            self.flag = 1
        else:
            self.flag = epoch

    def get_flag(self):
        return self.flag % 10

    def top_sim(self, sim_adj, toph, num_node):
        sim_node = torch.topk(sim_adj, k=toph+args.self_loop, dim=1)
        sim_node_value = sim_node.values[:, args.self_loop:].reshape((-1)) / toph
        sim_node_col = sim_node.indices[:, args.self_loop:].reshape((-1))
        sim_node_row = torch.tensor(range(num_node)).long().reshape((-1, 1)).to(device)
        sim_node_row = torch.tile(sim_node_row, [1, toph]).reshape((-1))
        sim_node_indices = torch.stack([sim_node_row, sim_node_col])
        sim_adj = torch.sparse.FloatTensor(sim_node_indices, sim_node_value, torch.Size(sim_adj.shape))
        return sim_adj

    def get_consin_adj(self, users_emb, items_emb):
        with torch.no_grad():
            user_sim = self.cosine_similarity(users_emb, users_emb)
            self.user_sim_adj = self.top_sim(user_sim, args.top_H, self.num_users)
            del user_sim
            item_sim = self.cosine_similarity(items_emb, items_emb)
            self.item_sim_adj = self.top_sim(item_sim, args.top_H, self.num_items)
            del item_sim

    def get_interaction_adj(self, emb_u, emb_i):
        with torch.no_grad():
            sim = torch.sigmoid(torch.matmul(emb_u, emb_i.transpose(0, 1)))
            adj = self.adj.to_dense()
            if args.dataset == 'Gowalla':
                l = 5
                num_b = sim.shape[0] // l
                if num_b * l < sim.shape[0]:
                    l = l + 1
                for i in range(l):
                    begin = i * num_b
                    end = (i + 1) * num_b
                    end = sim.shape[0] if end > sim.shape[0] else end
                    adj[begin:end] = (1-args.beta) * torch.mul(adj[begin:end], sim[begin:end]) + args.beta * adj[begin:end]
            else:
                adj = (1-args.beta) * adj * sim + args.beta * adj
        return adj

    def computer(self):
        emb_u = self.embedding_user.weight
        emb_i = self.embedding_item.weight
        adj = self.get_interaction_adj(emb_u, emb_i)
        users_embs = [emb_u]
        items_embs = [emb_i]
        for layer in range(args.layer):
            emb_u = torch.mm(adj, items_embs[-1])
            emb_i = torch.mm(adj.transpose(0, 1), users_embs[-1])
            users_embs.append(emb_u)
            items_embs.append(emb_i)
        del adj

        if self.get_flag() == 0:
            self.get_consin_adj(emb_u, emb_i)
        sim_emb_u = torch.sparse.mm(self.user_sim_adj, self.embedding_user.weight)
        sim_emb_i = torch.sparse.mm(self.item_sim_adj, self.embedding_item.weight)

        users_embs = torch.stack(users_embs, dim=1)
        items_embs = torch.stack(items_embs, dim=1)
        users_embs = torch.mean(users_embs, dim=1)
        items_embs = torch.mean(items_embs, dim=1)

        users_embs = users_embs + args.alpha * sim_emb_u
        items_embs = items_embs + args.alpha * sim_emb_i
        del sim_emb_i, sim_emb_u
        return users_embs, items_embs

    def getRating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = torch.matmul(users_emb, items_emb.t())
        return self.act(scores)

    def bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss_bpr = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        return loss_bpr

    def inference(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        loss_bpr = self.bpr_loss(users_emb, pos_emb, neg_emb)
        reg_loss = (1 / 2) * (torch.norm(self.embedding_user(users.long()), p=2).pow(2) +
                              torch.norm(self.embedding_item(pos_items.long()), p=2).pow(2) +
                              torch.norm(self.embedding_item(neg_items.long()), p=2).pow(2)) / float(len(users))

        return loss_bpr, reg_loss

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def train(n_user, m_item, num_traindata, traindata, testdata, model, group):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_results = []
    for g in range(len(group.keys()) + 1):
        best_results.append({'precision': np.zeros(len(topks)), 'recall': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))})
    max_value = 0.
    step = 0
    for epoch in range(args.epochs):
        model.update_flag(epoch)
        if epoch % 10 == 0:
            test_time = time()
            results = evaluate(n_user, m_item, traindata, testdata, model, group)
            if max_value <= results[-1]['recall'][0]:
                max_value = results[-1]['recall'][0]
                step = 0
                # torch.save(model.state_dict(), model_file)
            for g in range(len(group.keys()) + 1):
                for key, value in best_results[g].items():
                    best_results[g][key] = np.maximum(value, results[g][key])
            tst_log = f'[TEST {round(time()-test_time, 2)}s], {str(results[-1]).replace("array(","").replace(")", "")}'
            print(tst_log)
            _log.write(tst_log + '\n')
            _log.flush()

        start = time()
        model.train()
        S = sampling.sample_negative(n_user, m_item, num_traindata, traindata, 1)
        users = torch.Tensor(S[:, 0]).long().to(device)
        posItems = torch.Tensor(S[:, 1]).long().to(device)
        negItems = torch.Tensor(S[:, 2]).long().to(device)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        total_batch = len(users) // args.batch_size + 1
        aver_loss = 0.
        for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(minibatch(users, posItems, negItems, batch_size=args.batch_size)):
            optimizer.zero_grad()
            bpr_loss, reg_loss = model.inference(batch_users, batch_pos, batch_neg)
            loss = bpr_loss + reg_loss * args.decay
            loss.backward()
            optimizer.step()
            aver_loss += loss
        aver_loss = aver_loss / total_batch
        trn_log = f'EPOCH {epoch + 1}/{args.epochs} [{round(time()-start, 2)}s], loss: {aver_loss:.5f}'
        print(trn_log)
        _log.write(trn_log + '\n')
        if step > args.early_stop:
            break
        step += 1
    log_result = ''
    for g in range(len(group.keys())):
        print(f'{g+1}-group Best:', str(best_results[g]).replace('array(', '').replace(')', ''))
        log_result += f'{g+1}-group:' + str(best_results[g]).replace('array(', '').replace(')', '') + '\n'
    print(f'Best:', str(best_results[-1]).replace('array(', '').replace(')', ''))
    log_result += 'Best:' + str(best_results[-1]).replace('array(', '').replace(')', '') + '\n'
    _log.write(log_result)
    _log.flush()
    _log.close()

def evaluate(n_user, m_item, traindata, testdata, model, group):
    if n_user / 10 > 100:
        u_batch_size = 100
    else:
        u_batch_size = 10
    # eval mode with no dropout
    model = model.eval()
    max_K = max(topks)
    results = []
    for g in range(len(group.keys()) + 1):
        results.append({'precision': np.zeros(len(topks)), 'recall': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))})
    with torch.no_grad():
        user_gpu = torch.LongTensor(range(n_user)).to(device)
        item_gpu = torch.LongTensor(range(m_item)).to(device)
        ratings = model.getRating(user_gpu, item_gpu)

        for g in range(len(group.keys())):
            users = group[g+1]
            users_list = []
            rating_list = []
            groundTrue_list = []
            batch_num = len(users) // u_batch_size
            if batch_num * u_batch_size == len(users):
                total_batch = batch_num
            else:
                total_batch = batch_num + 1
            for batch_users in minibatch(users, batch_size=u_batch_size):
                allPos = getvalue(traindata, batch_users)
                groundTrue = getvalue(testdata, batch_users)
                rating = ratings[batch_users]
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            # print(rating_list)
            X = zip(rating_list, groundTrue_list)
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
            for result in pre_results:
                results[g]['recall'] += result['recall']
                results[g]['precision'] += result['precision']
                results[g]['ndcg'] += result['ndcg']
                results[-1]['recall'] += result['recall']
                results[-1]['precision'] += result['precision']
                results[-1]['ndcg'] += result['ndcg']
            results[g]['recall'] /= float(len(users))
            results[g]['precision'] /= float(len(users))
            results[g]['ndcg'] /= float(len(users))
        results[-1]['recall'] /= float(n_user)
        results[-1]['precision'] /= float(n_user)
        results[-1]['ndcg'] /= float(n_user)
        del ratings
        return results

def getvalue(x, indexes):
    values = []
    for i in indexes:
        values.append(x[i])
    return values

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall), 'precision': np.array(pre), 'ndcg': np.array(ndcg)}

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

if __name__ == '__main__':
    n_user, m_item, num_traindata, traindata, testdata, adj, group = load()
    model = MDGCF(n_user, m_item, adj).to(device)
    if args.task == 'train':
        train(n_user, m_item, num_traindata, traindata, testdata, model, group)
    if args.task == 'test':
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        results = evaluate(n_user, m_item, traindata, testdata, model, group)
        for g in range(len(group.keys())):
            print(f'{g + 1}-group:', str(results[g]).replace('array(', '').replace(')', ''))
        print(f'[TEST]:', str(results[-1]).replace('array(', '').replace(')', ''))
