import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import adjusted_rand_score

from a import centrality_cal, merge_sort_values, clustering, data_preprocess
from b import human_annotation
from dataset.zoo.zoo_generate import generate_zoo_data


def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec

def skeleton_process(Graph):
    clusters = []
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


def DS_reconstrction(outliers, new_edges, skeleton,real_labels):
    for node in outliers:
        anomaly = list(skeleton.out_edges(node))[0]
        skeleton.remove_edge(anomaly[0], anomaly[1])
    for edge in new_edges:
        skeleton.add_edge(edge[0], edge[1])
    predict_labels=skeleton_process(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    return ARI


def dsr(data, real_labels,budget,l):
    skeleton,representatives,uncertainty_score=clustering(data)
    skeleton=centrality_cal(skeleton)
    uncertainty_score = merge_sort_values(uncertainty_score)
    outliers,new_edges=human_annotation(data,skeleton,uncertainty_score,representatives,l,real_labels,budget)
    ARI=DS_reconstrction(outliers,new_edges,skeleton,real_labels)
    return ARI


if __name__ == '__main__':
    l=102
    budget = 80
    data, real_labels = data,labels=generate_zoo_data(path="dataset/zoo/zoo.data")
    ARI = dsr(data, real_labels, budget, l)
    print(ARI)
752