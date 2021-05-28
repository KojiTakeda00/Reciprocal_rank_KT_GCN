# -*- coding: utf-8 -*-
import numpy as np
import dgl
import pandas as pd
import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import networkx as nx
import os

sys.path.append("../paramdir")
import parameters

#from my_function_robotcar imporat *
args = sys.argv

#無向グラフ作成
def build_SYNTHIA_graph(num_node,edges):
    g=dgl.DGLGraph()
    g.add_nodes(num_node)
    if len(edges) != 0:
        g.add_edges(edges[:,0],edges[:,1])
    return g

#edgeなしの場合はこちらの関数を利用してgraph作成
def build_SYNTHIA_graph_without_edges(num_node,edges):
    g=dgl.DGLGraph()
    g.add_nodes(num_node)
    return g

#jsonふぁいる読み込み用の関数定義
import pickle
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


#multi view scene graphの画像枚数
frames = parameters.frames
#frames = 3

#multi view scene graphの画像間の距離
distance_thresh = parameters.distance_between_frames






train_date = parameters.train_date     #train dataの日付
test_date = parameters.test_date      #test dataの日付

class_list_train = np.loadtxt("../class_list/train_" + train_date + "_test_" + test_date + "/class_list_tr\
ain_"+train_date+".txt",str)
class_list_test = np.loadtxt("../class_list/train_" + train_date + "_test_" + test_date + "/class_list_tes\
t_"+test_date +".txt",str)

classes=len(set(class_list_train[:,1]))
input_dim = classes

test_result = np.loadtxt("test_result_Raw.txt")
test_result = test_result.T

train_result = np.loadtxt("train_result_Raw.txt")
train_result = train_result.T

#gps_test = np.loadtxt("/share_dir/hdd1/takeda/RobotCarDataset-Scraper/Downloads/" + test_date + "/interporated_gps.txt")
#gps_test_str = np.loadtxt("/share_dir/hdd1/takeda/RobotCarDataset-Scraper/Downloads/" + test_date + "/interporated_gps.txt",str)
#gps_train = np.loadtxt("/share_dir/hdd1/takeda/RobotCarDataset-Scraper/Downloads/" + train_date + "/interporated_gps.txt")
#gps_train_str = np.loadtxt("/share_dir/hdd1/takeda/RobotCarDataset-Scraper/Downloads/" + train_date + "/interporated_gps.txt",str)


test_result_canny = np.loadtxt("test_result_canny.txt")
test_result_canny = test_result_canny.T

train_result_canny = np.loadtxt("train_result_canny.txt")
train_result_canny = train_result_canny.T

test_result_semantic = np.loadtxt("test_result_semantic.txt")
test_result_semantic = test_result_semantic.T

train_result_semantic = np.loadtxt("train_result_semantic.txt")
train_result_semantic = train_result_semantic.T

 

gps_train_dict = pickle_load("gps_dict/"+train_date+"_gps.pickle")
gps_test_dict = pickle_load("gps_dict/"+test_date+"_gps.pickle")
#format of this dictionary is dict["image_name"] = [latitude, longitude]


def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f


def inverse_rank(x):
    argsorted = np.argsort(x,axis = 1)
    x = np.zeros((argsorted.shape))
    for idx,row in enumerate(argsorted):
        for i,args in enumerate(row):
            x[idx][args] = 1./(i + 1)
    return x



train_rank_canny = inverse_rank(train_result_canny)
test_rank_canny = inverse_rank(test_result_canny)

train_rank_semantic = inverse_rank(train_result_semantic)
test_rank_semantic = inverse_rank(test_result_semantic)





####train graph generation###
train_graph_list = []
train_rank = inverse_rank(train_result)



"""
###process sequence###
for path_and_label,feature_array in tqdm(zip(class_list_train,train_rank)):
    #print(semantic_npy_path)
    png_name = path_and_label[0]
    #print(png_name)

    edge_list = [np.array([0,0])]
    num_node= 1

    G=build_SYNTHIA_graph(num_node,np.array(edge_list))
    
    G.ndata['feat'] = torch.from_numpy(np.array([feature_array]).astype(np.float32)).clone()
    #get label
    
    label = int(path_and_label[1])
    train_graph_list.append([G,int(label)])
"""


def calc_dis(lat_diff,lon_diff):
    lat_meter_robotcar = 70197.1396 #meter[51 degree]
    lon_meter_robotcar = 111248.2712 #meter[51 degree]
    #computed by https://www.movable-type.co.uk/scripts/latlong.html
    return np.linalg.norm(np.array([lat_diff*lat_meter_robotcar,lon_diff*lon_meter_robotcar]), ord=2)


def calc_distance_from_gps_coodinate_robotcar(coo1,coo2):
    #coodinate must be latitude,longtitude
    lat_diff = coo1[0] - coo2[0]
    lon_diff = coo1[1] - coo2[1]
    lat_meter_robotcar = 70197.1396 #meter[51 degree]
    lon_meter_robotcar = 111248.2712 #meter[51 degree]
    return np.linalg.norm(np.array([lat_diff*lat_meter_robotcar,lon_diff*lon_meter_robotcar]), ord=2)



#print(calc_distance_from_gps_coodinate_robotcar(gps_test[1000][1:],gps_test[1005][1:]))



#gps_test_coodinates = gps_test
#gps_train_coodinates = gps_train




###process sequence###
for i,[path_and_label,feature_array] in tqdm(enumerate(zip(class_list_train,train_rank))):
    #print(semantic_npy_path)

    
    
    edge_list = []
    
    

    now = i
    over_idx = 0
    label_start_frame = path_and_label[1]


    node_idx_list = [now]

    num_name_list = [class_list_train[now][0].split(".")[0]]
    feature_list = [train_rank[now]]
    
    for idx in range(frames - 1):
        for distance_idx in range(1,100):
            if now + distance_idx >= len(class_list_train)-1 or class_list_train[now][1] != class_list_train[now + distance_idx][1]:
                over_idx = 1
                break

            image_num_now = class_list_train[now][0].split(".")[0]
            gps_coo_now = gps_train_dict[image_num_now]
            image_num_future = class_list_train[now + distance_idx][0].split(".")[0]
            gps_coo_future = gps_train_dict[image_num_future]
            distance = calc_distance_from_gps_coodinate_robotcar(gps_coo_now, gps_coo_future)
            #print(distance)
            if distance > distance_thresh:
                #print(distance_idx)
                break
        if over_idx == 1:
            break
        feature_list.append(train_rank[now + distance_idx])
        node_idx_list.append(now + distance_idx)
        num_name_list.append(image_num_future)
        now = now + distance_idx

        #self edges
        #edge_list.append(np.array([idx,idx]))
        
        
        edge_list.append(np.array([idx,idx + 1]))
        edge_list.append(np.array([idx + 1,idx]))

        
    if over_idx == 1:
        continue
    
    #png_name = path_and_label[0]
    #print(png_name)

    num_node= frames * 3
    idx += 1
    for idx,eimage_idx in enumerate(node_idx_list):
        feature_list.append(train_rank_canny[eimage_idx])
        edge_list.append(np.array([idx,idx + frames]))
        edge_list.append(np.array([idx + frames,idx]))
    for idx,eimage_idx in enumerate(node_idx_list):
        feature_list.append(train_rank_semantic[eimage_idx])
        edge_list.append(np.array([idx,idx + frames*2]))
        edge_list.append(np.array([idx + frames*2,idx]))
        

    
    

    G=build_SYNTHIA_graph(num_node,np.array(edge_list))
    
    G.ndata['feat'] = torch.from_numpy(np.array(feature_list).astype(np.float32)).clone()
    #get label
    
    label = int(path_and_label[1])
    train_graph_list.append([G,int(label)])

    #visualize_graph
    
    #nx.draw(G.to_networkx(), with_labels=True)
    #plt.savefig("graph.png")
    
    #print(G.ndata["feat"].shape)
    #print(G.edata)
    #sys.exit()
    #break

trainset = train_graph_list

epochs = 5
#epochs = 10


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    
    accum = torch.sum(nodes.mailbox['m'], 1)
    #print(accum)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')







class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.ndata['feat']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)





# Create training and test sets.

#trainset = MiniGCDataset(320, 10, 20)
#testset = MiniGCDataset(80, 10, 20)

#trainset=graph_lst
#testset=graph_lst






data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)



# Create model
model = Classifier(input_dim, 256, classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)




####test graph generation###
test_graph_list = []
test_rank = inverse_rank(test_result)


sequence_list = []

###process sequence###
for i,[path_and_label,feature_array] in tqdm(enumerate(zip(class_list_test,test_rank))):
    #print(semantic_npy_path)

    num_node= frames
    
    edge_list = []
    


    now = i
    over_idx = 0
    label_start_frame = path_and_label[1]

    node_idx_list = [now]
    num_name_list = [class_list_test[now][0].split(".")[0]]
    feature_list = [test_rank[now]]
    
    for idx in range(frames - 1):
        for distance_idx in range(1,100):
            if now + distance_idx >= len(class_list_test)-1 or class_list_test[now][1] != class_list_test[now + distance_idx][1]:
                over_idx = 1
                break

            image_num_now = class_list_test[now][0].split(".")[0]
            gps_coo_now = gps_test_dict[image_num_now]
            image_num_future = class_list_test[now + distance_idx][0].split(".")[0]
            gps_coo_future = gps_test_dict[image_num_future]
            distance = calc_distance_from_gps_coodinate_robotcar(gps_coo_now, gps_coo_future)
            #print(distance)
            if distance > distance_thresh:
                #print(distance_idx)
                break
        if over_idx == 1:
            break
        node_idx_list.append(now + distance_idx)
        feature_list.append(test_rank[now + distance_idx])
        num_name_list.append(image_num_future)
        now = now + distance_idx

        #self edges
        #edge_list.append(np.array([idx,idx]))
        
        
        edge_list.append(np.array([idx,idx + 1]))
        edge_list.append(np.array([idx + 1,idx]))
    if over_idx == 1:
        continue
    
    #png_name = path_and_label[0]
    #print(png_name)
    num_node= frames * 3
    idx += 1
    for idx,eimage_idx in enumerate(node_idx_list):
        feature_list.append(test_rank_canny[eimage_idx])
        edge_list.append(np.array([idx,idx + frames]))
        edge_list.append(np.array([idx + frames,idx]))
    for idx,eimage_idx in enumerate(node_idx_list):
        feature_list.append(test_rank_semantic[eimage_idx])
        edge_list.append(np.array([idx,idx + frames*2]))
        edge_list.append(np.array([idx + frames*2,idx]))

    
    sequence_list.append(num_name_list)

    G=build_SYNTHIA_graph(num_node,np.array(edge_list))
    
    G.ndata['feat'] = torch.from_numpy(np.array(feature_list).astype(np.float32)).clone()
    #get label
    
    label = int(path_and_label[1])
    test_graph_list.append([G,int(label)])

    #sys.exit()
    #break



#np.savetxt("result_sequencial/sequence_visualize_with_edges.txt",np.array(sequence_list),fmt = "%s")


testset = test_graph_list





model.eval()
# Convert a list of tuples to two lists
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))


numpy_probs_Y = probs_Y.to('cpu').detach().numpy()

os.makedirs("result_sequencial",exist_ok = True)


np.savetxt("result_sequencial/prediction_" + str(epochs) + "_RGB_canny_sum-aggregation.txt",numpy_probs_Y)


numpy_test_Y = test_Y.to('cpu').detach().numpy()
np.savetxt("result_sequencial/GT_" + str(epochs) + "_RGB_canny_sum-aggregation.txt",numpy_test_Y)

