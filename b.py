import numpy as np
from scipy.spatial.distance import euclidean




def connections_cal(data, node, neighborhoods):
    connections = []
    for i in range(len(neighborhoods)):
        distances = []
        for neighbor in neighborhoods[i]:
            distances.append(euclidean(data[node], data[neighbor]))
        index = np.argmin(distances)
        connections.append([node, neighborhoods[i][index], distances[index],i])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections



def interaction_process(connections,skeleton, real_labels, neighborhoods, neighborhoods_behind, count, l,budget,new_edges):
    flag = False
    for i in range(len(connections)):
        node1 = int(connections[i][0])
        node2 = int(connections[i][1])
        neighborhood_index = int(connections[i][3])
        if real_labels[node1] == real_labels[node2]:
            count=count+1
            if count > budget:
                return neighborhoods, neighborhoods_behind, count, new_edges
            new_edges.append([node1,node2])
            flag = True
            if len(neighborhoods[neighborhood_index]) < l:
                neighborhoods[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['centrality']<skeleton.nodes[neighborhoods_behind[neighborhood_index][0]]['centrality']:
                    neighborhoods_behind[neighborhood_index]=[node1]
            if len(neighborhoods[neighborhood_index]) >= l:
                if skeleton.nodes[node1]['centrality'] > skeleton.nodes[neighborhoods_behind[neighborhood_index][0]]['centrality']:
                    neighborhoods[neighborhood_index].remove(neighborhoods_behind[neighborhood_index][0])
                    neighborhoods[neighborhood_index].append(node1)
                    a = []
                    for j in neighborhoods[neighborhood_index]:
                        a.append(skeleton.nodes[j]['centrality'])
                    c = neighborhoods[neighborhood_index][np.argmin(a)]
                    neighborhoods_behind[neighborhood_index] = [c]
            break
        if real_labels[node1] != real_labels[node2]:
            count = count + 1
            if count > budget:
                return neighborhoods, neighborhoods_behind, count, new_edges
    if flag == False:
        neighborhoods.append([node1])
        neighborhoods_behind.append([node1])
    return neighborhoods, neighborhoods_behind, count,new_edges




def human_annotation(data,skeleton,uncertainty_score,representatives,l,real_labels,budget):
    # 创建neighborhoods
    neighborhoods=[representatives]
    neighborhoods_behind=[representatives]
    count=0
    new_edges=[]
    outliers=[]
    for node in uncertainty_score:

        connections=connections_cal(data,node,neighborhoods)
        neighborhoods, neighborhoods_behind, count,new_edges = interaction_process(connections,skeleton, real_labels, neighborhoods, neighborhoods_behind, count, l,budget,new_edges)
        if count>budget:
            break
        outliers.append(node)
    return outliers,new_edges


