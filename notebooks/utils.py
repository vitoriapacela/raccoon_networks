import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.metrics import silhouette_score
from os import listdir, chdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
from networkx import edge_betweenness_centrality
from random import random
from itertools import islice
from networkx.algorithms.community.centrality import girvan_newman

## Hierarchical clustering
def d_mat(g):
    """Calculates the custom distance 1/(1 + log(interaction + 1))
        as a matrix for a graph

    Input:  Graph object g

    Output: Corrected log(s + 1) of interaction matrix"""
    A = nx.to_numpy_matrix(g)
    A = 1 / (1 + np.log(A + 1))
    np.fill_diagonal(A, 0)
    return A


def hc_cluster_score(X, g):
    """Calculates the silhouette score for all the possible k clusters defined in Hierarchical Clustering.

    Input:  Graph object g
            Distance matrix X

    Output: Dict with labels of the cluster with the best score
            List with the silhouette scores"""
    scores = []
    labels = []
    for i in range(2, len(X)):
        hierarchical_model = AC(n_clusters=i, affinity='precomputed', linkage='average').fit(X)
        l = hierarchical_model.labels_
        s = silhouette_score(X, l, metric="precomputed")
        scores.append(s)
        labels.append(l)

    idx = np.argmax(scores)
    clust_lab = {list(g.nodes())[i]: list(labels[idx])[i] for i in range(len(labels[idx]))}
    return clust_lab, scores


def create_edges(G, pos):
    """Creates a trace with the edges of a graph

    Input:  Graph object G
            Positional dict pos

    Output: Trace of edges"""
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    return edge_trace


def create_nodes(G, pos):
    """Creates a trace with the nodes of a graph

    Input:  Graph object G
            Positional dict pos

    Output: Trace of nodes with color given by cluster_label """
    node_x = [-1]
    node_y = [-1]
    for p in pos.values():
        x, y = p
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(size=20,
                    line_width=2))

    node_adjacencies = [-1]
    node_text = ['na']
    c, xx = cluster_score(d_mat(G), G)
    for k in pos.keys():
        if k in G.nodes:
            node_adjacencies.append(c[k])
            node_text.append('Cluster: ' + str(c[k]))
        else:
            node_adjacencies.append(-1)
            node_text.append('Cluster: NA')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    return node_trace

## K-means
def preprocess(g):
    """ Filters the desired nodes, creates adjacency matrix based on the interaction time metric.
    
    Input: g (graph object)
    Output: Interaction matrix X (numpy matrix)
    """
    # Filtering: remove edges > 15
    remove = [node for node in g.nodes if int(node) > 15]
    g.remove_nodes_from(remove)

    # the distance is the inverse of the interaction
    # take log because the interaction values have a big range
    A = nx.to_numpy_matrix(g)
    X = 1 / (1 + np.log(A + 1))
    np.fill_diagonal(X, 0)
    return X


def plot_spring(g, labels):
    """ Plots graph in spring layout.
    Input: g (graph object)
           labels (list of cluster IDs)
    """
    plt.figure()
    ax1 = plt.subplot(111)

    gn_pos = nx.spring_layout(g)
    g1 = nx.draw_networkx_nodes(g, gn_pos, node_color=labels, node_size=500, label=labels)
    nx.draw_networkx_edges(g, gn_pos, alpha=0.2)

    plt.axis('off')
    legend1 = ax1.legend(*g1.legend_elements(), title="Classes")
    ax1.add_artist(legend1)
    plt.show()

def km_cluster_score(X, g):
    """
    Calculates the silhouette score for all the possible k clusters defined in K-Means Clustering.

    Input:  Graph object g
            Distance matrix X

    Output: Dict with labels of the cluster with the best score
            List with the silhouette scores
    """
    scores = []
    labels = []
    for i in range(2, len(X)):
        kmeans_model = KMeans(n_clusters=i, precompute_distances=True).fit(X)
        l = kmeans_model.labels_
        s = silhouette_score(X, l, metric="precomputed")
        scores.append(s)
        labels.append(l)

    idx = np.argmax(scores)
    clust_lab = {list(g.nodes())[i]: list(labels[idx])[i] for i in range(len(labels[idx]))}
    return clust_lab, scores

def km_cluster_score_full(X, g):
    """
    Calculates the silhouette score for all the possible k clusters defined in K-Means Clustering. Plots the clusters for every week iteration.
    
    Input:  Graph object g
            Distance matrix X
            
    Output: List of the clusters with best score
            List with the best silhouette scores
            Numpy array with all the silhouette scores for all the weeks
            Numpy array with all the clusters for all the weeks
    """
    scores = []
    labels = []
    for i in range(2, len(X)):
        kmeans_model = KMeans(n_clusters=i, precompute_distances=True).fit(X)
        l = kmeans_model.labels_
        s = silhouette_score(X, l, metric="precomputed")
        scores.append(s)
        labels.append(l)

    idx = np.argmax(scores)
    clust_lab = labels[idx]
    plot_spring(g, clust_lab)
    return clust_lab, scores[idx], np.asarray(scores), np.asarray(labels)

## Centralities
def weight(G, node):
    """Calculates the total of interaction of a single raccoon

    Input:  Graph object G
            Raccoon_ID node

    Output: Corrected log(s + 1) of interaction"""
    s = 0
    for g in G[node]:
        if g in filt:
            s += G[node][g]['weight']
    return np.log(s + 1)

## Communities
def mve(G):
    """
    Finds the Most Valuable Edge.
    
    Input:  Graph object G
    Output: List with the most valuable edge's node and weight.
    """
    # assume that the graph is already filtered
    mx = 0 
    for n in G.nodes():
        for w in G[n]:
            if G[n][w]['weight'] > mx:
                mx = G[n][w]['weight']
                edge = (n, w)
    return edge

def comm(g, num_iter=1):
    """
    Community clustering with Girvan-Newman's algorithm.
    
    Input:  Graph object g
            (int) Number of iterations num_iter
    Output: Dict with cluster labels
    """
    gn_generator = girvan_newman(g, mve)
    gn_communities = next(islice(gn_generator, num_iter, None))

    gn_dict_communities = {}

    for i, c in enumerate(gn_communities):
        for node in c:
            gn_dict_communities[node] = i + 1

    for node in g:
        if node not in gn_dict_communities.keys():
            gn_dict_communities[node] = -1

    return gn_dict_communities

def cluster_score_comm(g, X, i):
    """Calculates the silhouette score for all the possible communities
    
    Input:  Graph object g
            Distance matrix X
            Number of iterations i
           
    Output: List with the silhouette scores"""
    l = np.array(list(comm(g, i).values()))
    if l[-1]==len(g):
        l[-1]=len(g) - 1
    s = silhouette_score(X, l, metric="precomputed")
    return s

def comm_auto(g, X):
    """Calculates the silhouette score for all iterations of the Girvan-Newman algorithm
    
    Input:  Graph object g
            Distance matrix X
           
    Output: Dict with labels of the cluster with the best score
            List with the silhouette scores"""
    n = 0
    i = 1
    max_score = -1
    while n<len(g):
        score = cluster_score_comm(g, X, i)
        if score>max_score:
            max_i = i
            max_score = score
        n = max(comm(g, i).values())
        i +=1
    return comm(g, max_i)