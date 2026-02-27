import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from networkx.drawing.nx_pydot import graphviz_layout
np.random.seed(0)

def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos,alpha=0.5, node_color="blue", with_labels=True,font_size=20,node_size=30)
    plt.axis("equal")
    plt.show()


def nearest_neighbor_cal(feature_space):
    neighbors=NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance,nearest_neighbors= neighbors.kneighbors(feature_space,return_distance=True)
    distance=distance[:,1]
    nearest_neighbors=nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance[i])
    return nearest_neighbors


def sub_nodes_cal(sub_S):
    points=None
    for edge in sub_S.edges:
        if sub_S.has_edge(edge[1],edge[0]):
            point1=edge[0]
            point2=edge[1]
            points = [point1, point2]
            break
    return points

def representative_find_sitation_2(points,skeleton):
    sum1 = 0
    in_edges = skeleton.in_edges(points[0])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum1 = sum1 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    sum2 = 0
    in_edges = skeleton.in_edges(points[1])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum2 = sum2 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    index = np.argmax([sum1, sum2])
    representative = points[index]
    return index,representative


def clustering_loop(feature_space,dict_mapping,skeleton,data,uncertainty_score):
    representatives = []
    edges=nearest_neighbor_cal(feature_space)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
        uncertainty=edges[i][2]
        skeleton.add_edge(edges[i][0],edges[i][1])
        uncertainty_score[edges[i][0]]=uncertainty
    Graph_t=nx.DiGraph()
    Graph_t.add_weighted_edges_from(edges)
    S_t = [Graph_t.subgraph(c).copy() for c in nx.weakly_connected_components(Graph_t)]
    for S_st in S_t:
        all_nodes = list(S_st.nodes)
        rnn=sub_nodes_cal(S_st)
        sums=[]
        for i in range(len(rnn)):
            sum=0
            for j in range(len(all_nodes)):
                sum=sum+distance.euclidean(data[rnn[i]], data[all_nodes[j]])
            sums.append(sum)
        index=np.argmin(sums)
        representative=rnn[index]
        representatives.append(representative)
        edge_remove=[rnn[index],rnn[1-index],skeleton[rnn[index]][rnn[1-index]]]
        skeleton.remove_edge(edge_remove[0], edge_remove[1])
    dict_mapping={}
    for i in range(len(representatives)):
        dict_mapping[i]=representatives[i]

    return representatives,skeleton,dict_mapping,uncertainty_score



def clustering(data):
    data = data_preprocess(data)
    feature_space=copy.deepcopy(data)
    dict_mapping = {}
    uncertainty_score={}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.DiGraph()
    while (True):
        representatives,skeleton,dict_mapping,uncertainty_score=clustering_loop(feature_space, dict_mapping,skeleton,data,uncertainty_score)
        feature_space=data[representatives]
        if len(representatives) == 1:
            break
    uncertainty_score.pop(representatives[0])
    return skeleton,representatives,uncertainty_score


def data_preprocess(data):
    size=np.shape(data)
    random_matrix=np.random.rand(size[0],size[1]) * 0.000001
    data=data+random_matrix
    return data

def centrality_cal(skeleton):
    points=list(skeleton.nodes)
    for i in range(len(points)):
        centrality=skeleton.in_degree(points[i])
        skeleton.nodes[points[i]]['centrality']=centrality
    return skeleton


def merge_sort_values(input_dict):
    items = list(input_dict.items())
    def merge_sort(items):
        if len(items) > 1:
            mid = len(items) // 2
            left_half = items[:mid]
            right_half = items[mid:]
            merge_sort(left_half)
            merge_sort(right_half)
            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                if left_half[i][1] > right_half[j][1]:
                    items[k] = left_half[i]
                    i += 1
                else:
                    items[k] = right_half[j]
                    j += 1
                k += 1
            while i < len(left_half):
                items[k] = left_half[i]
                i += 1
                k += 1
            while j < len(right_half):
                items[k] = right_half[j]
                j += 1
                k += 1
    merge_sort(items)
    sorted_keys = [item[0] for item in items]
    return sorted_keys









