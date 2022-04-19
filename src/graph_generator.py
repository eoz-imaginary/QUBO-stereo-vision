import networkx as nx
import numpy as np
import random


def gen_digraph_5():
    test_e_lst = [('s', 0, {"weight": 1}), ('s', 1, {"weight": 1}), (0, 2, {"weight": 3}), (0, 't', {"weight": 2}),
                  (1, 't', {"weight": 1})]
    test_nodes = [0, 1, 2, 's', 't']
    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(test_nodes)
    nx_g.add_edges_from(test_e_lst)
    return nx_g, test_e_lst

def gen_digraph_20():
    nodes = list(range(0, 20))
    nodes[-2] = 's'
    nodes[-1] = 't'
    edges = [('s', 0, {'weight': 1}),
             ('s', 1, {'weight': 2}),
             ('s', 2, {'weight': 3}),
             ('s', 3, {'weight': 4}),
             (0, 4, {'weight': 5}),
             (4, 8, {'weight': 5}),
             (8, 12, {'weight': 5}),
             (1, 5, {'weight': 5}),
             (5, 9, {'weight': 5}),
             (9, 13, {'weight': 5}),
             (2, 6, {'weight': 5}),
             (6, 10, {'weight': 5}),
             (10, 14, {'weight': 5}),
             (3, 7, {'weight': 5}),
             (7, 11, {'weight': 5}),
             (11, 15, {'weight': 5}),
             (3, 16, {'weight': 5}),
             (7, 17, {'weight': 5}),
             (11, 18, {'weight': 5}),
             (16, 17, {'weight': 5}),
             (17, 18, {'weight': 5}),
             (18, 15, {'weight': 5}),
             (12, 't', {'weight': 1}),
             (13, 't', {'weight': 2}),
             (14, 't', {'weight': 3}),
             (15, 't', {'weight': 4}),
        ]
    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(nodes)
    nx_g.add_edges_from(edges)
    return nx_g, edges


def gen_digraph_30():
    nodes = list(range(0, 20))
    nodes[-2] = 's'
    nodes[-1] = 't'
    edges = [('s', 0, {'weight': 4}),
             ('s', 28, {'weight': 4}),
             ('s', 27, {'weight': 4}),
             ('s', 26, {'weight': 4}),
             ('s', 25, {'weight': 4}),
             (0, 1, {'weight': 7}),
             (1, 2, {'weight': 7}),
             (2, 3, {'weight': 7}),
             (3, 4, {'weight': 7}),
             (4, 5, {'weight': 7}),
             (5, 6, {'weight': 7}),
             (6, 7, {'weight': 7}),
             (7, 8, {'weight': 7}),
             (8, 9, {'weight': 7}),
             (9, 10, {'weight': 7}),
             (10, 11, {'weight': 7}),
             (11, 12, {'weight': 7}),
             (12, 13, {'weight': 7}),
             (13, 14, {'weight': 7}),
             (14, 15, {'weight': 7}),
             (15, 16, {'weight': 7}),
             (16, 17, {'weight': 7}),
             (17, 18, {'weight': 7}),
             (18, 19, {'weight': 7}),
             (19, 20, {'weight': 7}),
             (20, 21, {'weight': 7}),
             (21, 22, {'weight': 7}),
             (22, 23, {'weight': 7}),
             (23, 24, {'weight': 7}),
             (24, 25, {'weight': 7}),
             (25, 26, {'weight': 7}),
             (26, 27, {'weight': 7}),
             (27, 28, {'weight': 7}),
             (28, 0, {'weight': 7}),
             (0, 13, {'weight': 3}),
             (28, 9, {'weight': 3}),
             (27, 10, {'weight': 3}),
             (26, 11, {'weight': 3}),
             (25, 12, {'weight': 3}),
             (9, 't', {'weight': 4}),
             (10, 't', {'weight': 4}),
             (11, 't', {'weight': 4}),
             (12, 't', {'weight': 4}),
             (13, 't', {'weight': 4}),
    ]

    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(nodes)
    nx_g.add_edges_from(edges)
    return nx_g, edges



