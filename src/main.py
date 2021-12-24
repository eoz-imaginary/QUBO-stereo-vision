from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import neal
import networkx as nx
from collections import defaultdict
import numpy as np
import argparse
import cv2
import maxflow as mf
import matplotlib.pyplot as plt
import csv
from src.graph import kolmogorov_graph, handle_infinities, handle_infinities_scale_color
from itertools import combinations
import sys


#takes in an image and splits it into patches
def get_image_patches(img, patch_w, patch_h):
    #convert to grayscale for now
    image = cv2.imread(img)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patch = gray_img[0:patch_w, 0:patch_h]
    #print(patch.shape)
    return patch

def plot_graph_3d(graph, nodes_shape, plot_terminal=True, plot_weights=True, font_size=7):

    w_h = nodes_shape[1] * nodes_shape[2]
    X, Y = np.mgrid[:nodes_shape[1], :nodes_shape[2]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: v for i, v in enumerate(aux)}

    for i in range(1, nodes_shape[0]):
        for j in range(w_h):
            positions[w_h * i + j] = [positions[j][0] + 0.3 * i, positions[j][1] + 0.2 * i]

    positions['s'] = np.array([-1, nodes_shape[1] / 2.0 - 0.5])
    positions['t'] = np.array([nodes_shape[2] + 0.2 * nodes_shape[0], nodes_shape[1] / 2.0 - 0.5])

    nxg = graph.get_nx_graph()
    if not plot_terminal:
        nxg.remove_nodes_from(['s', 't'])

    nx.draw(nxg, pos=positions)
    nx.draw_networkx_labels(nxg, pos=positions)
    if plot_weights:
        edge_labels = dict([((u, v), d['weight']) for u, v, d in nxg.edges(data=True)])
        nx.draw_networkx_edge_labels(nxg,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)
    plt.axis('equal')
    plt.savefig('testgraph55.png')

#old version
def create_graph(img, width, height, depth):
    g = mf.GraphInt()
    nodeids = g.add_grid_nodes((depth, height, width))
    structure = np.array([[0, 1, 0,
                           1, 0, 1,
                           0, 1, 0]])
    g.add_grid_edges(nodeids, weights=img, structure=structure)

    #img pixels are source caps, inverse img pixels are sink caps
    g.add_grid_tedges(nodeids[:, :, 0], img, 255-img)
    #add sink
    g.add_grid_tedges(nodeids[:, :, -1], img, 255-img)

    return nodeids, g

# from https://github.com/dwave-examples/image-segmentation/blob/main/image_segmentation.py
def unindexing(a, img):
    rows, cols = img.shape
    print(a, cols)
    y1 = a % cols
    x1 = int(a/cols)
    return (x1, y1)

def compute_alpha(g, edge_list):
    # --> formula for alpha <--
    # Sum_over_edges [cost{u, v}] - Sum_over_nodes [cost{s, u} + cost{t, u}]

    a_term1 = 0
    a_term2 = 0

    for i in range(len(edge_list)):
        uv_weight = edge_list[i][2]['weight']
        a_term1 += uv_weight

    for i in range(len(edge_list)):
        u = edge_list[i][0]
        v = edge_list[i][1]

        u_in = list(g.in_edges(u, data=True))
        u_out = list(g.out_edges(u, data=True))
        src_edge_ind = None
        sink_edge_ind = None


        if u != ('s' or 't'):
            for j in range(len(u_in)):
                if u_in[j][0] == 's':
                    #print("u_in:", u_in)
                    src_edge_ind = j

            for k in range(len(u_out)):
                if u_out[k][1] == 't':
                    #print("u_out:", u_out)
                    sink_edge_ind = k

        if (src_edge_ind is not None) and (sink_edge_ind is not None):
            #print("neither ind is none")

            print(u_in[src_edge_ind])
            print(u_out[sink_edge_ind])
            a_term2 += u_in[src_edge_ind][2]['weight'] + u_out[sink_edge_ind][2]['weight']

    return a_term1 - a_term2

def main():

    # parse input images

    #TODO: take images as cmd arguments
    left = "images/middlebury_flowerpots_view1.png"
    right = "images/middlebury_flowerpots_view2.png"
    l_test = cv2.cvtColor(cv2.imread(left), cv2.COLOR_RGB2GRAY)
    l_test_sm = cv2.resize(l_test, (0, 0), fx=0.01, fy=0.01)
    r_test = cv2.cvtColor(cv2.imread(right), cv2.COLOR_RGB2GRAY)
    r_test_sm = cv2.resize(r_test, (0, 0), fx=0.01, fy=0.01)
    #l_patch = get_image_patches(left, 15, 15)
    #r_patch = get_image_patches(right, 15, 15)
    #print(r_test_sm.shape)

    g = kolmogorov_graph(l_test_sm, r_test_sm)
    nx_g = g.get_nx_graph()


    Q_size = (2*len(nx_g.edges())) + len(nx_g.nodes())
    vert_ind_start = 0
    y_ind_start = vert_ind_start + len(nx_g.nodes())
    w_ind_start = y_ind_start + len(nx_g.edges())
    print(Q_size)
    print("y start = ", y_ind_start, "w start = ", w_ind_start, "for a graph of ", len(nx_g.nodes()), " nodes and ", len(nx_g.edges()), " edges.")
    Q = np.zeros((Q_size, Q_size))
    #IN NODE LIST: src at index len(..)-2, sink at len(..)-1
    src_ind = len(nx_g.nodes())-2
    sink_ind = len(nx_g.nodes())-1
    e_lst = list(nx_g.edges(data=True))

    node_lst = list(nx_g.nodes)
    test_node = e_lst[9]
    #print(test_node)
    #print("indexing?", node_lst[test_node])
    #print("src?", node_lst[src_ind])
    #print("sink?", node_lst[sink_ind])

    alpha = compute_alpha(nx_g, e_lst)
    print("alpha is: ", alpha)

    # H_COST
    for i in range(len(e_lst)):
        u = e_lst[i][0]
        v = e_lst[i][1]
        weight = e_lst[i][2]['weight']
        x = y_ind_start + i
        y = y_ind_start + i
        Q[x, y] += weight
        if (x or y) > w_ind_start:
            print("DISASTER!! y_uv at index reserved for w_uv")
            print("x = ", x)
            print("y = ", y)
            print("i = ", i)


    # -----> H_QUBO_Penalty <-----
    # --> first 'half' <--
    # alpha*(1 - x_s - x_t + 2 * x_s * x_t)
    #TODO: might need to add offset of -1 to all cells?
    Q[src_ind, src_ind] += -1*alpha
    Q[sink_ind, sink_ind] += -1*alpha
    Q[src_ind, sink_ind] += 2*alpha

    # --> second 'half' <--
    # alpha * Sum_over_edges [x_u + x_v + 2w_uv + x_u * y_uv + x_v * y_uv - 2x_u * w_uv - 2x_v * w_uv - 2y_uv * w_uv]
    for i in range(len(e_lst)):
        # set up Q-matrix indicies

        u = e_lst[i][0]
        v = e_lst[i][1]

        if u == 's':
            u = src_ind
        if v == 't':
            v = sink_ind

        # adding vert_ind_start in next two lines unecessary, but there for clarity
        x_u = vert_ind_start + u
        x_v = vert_ind_start + v

        y_uv = y_ind_start + i
        w_uv = w_ind_start + i

        # add values to Q-matrix
        Q[x_u, x_u] += 1*alpha
        Q[x_v, x_v] += 1*alpha
        Q[w_uv, w_uv] += 2*alpha
        Q[x_u, y_uv] += 1*alpha
        Q[x_v, y_uv] += 1*alpha
        Q[x_u, w_uv] += -2*alpha
        Q[x_v, w_uv] += -2*alpha
        Q[y_uv, w_uv] += -2*alpha

    '''
    for u, v in nx_g.edges():
        #issue with s & t labels
        u_before = u
        v_before = v
        try:
            if u == 's':
                u = src_ind
            if u == 't':
                u = sink_ind
            if v == 's':
                v = src_ind
            if v == 't':
                v = sink_ind
            Q[y_ind_start+u, y_ind_start+v] += nx_g.edges[u_before, v_before]['weight']
        except KeyError:
            print("u, v before:", u_before, v_before)
            print("u, v after:", u, v)
            print("edge using old indicies: ", nx_g.edges[u_before, v_before])
            print(len(nx_g.nodes()) - 2, len(nx_g.nodes()) - 1)
            print(nx_g.edges[u, v])
    '''
    #H_QUBO_PENALTY




    print(Q.shape)
    print(Q)

    # Write to CSV to examine (DEBUGGING ONLY!!)
    '''
    with open("Q.csv", "w+") as file:
        csvWriter = csv.writer(file, delimiter=",")
        csvWriter.writerows(Q)
    '''


    ''' 
    #OLD QUBO 
    alpha = 4
    Q = defaultdict(int)

    #H_cost = sum {y_uv * edge_cost(u, v)}
    #y_uv = 1 iff (u, v) in cut, else 0
    # (i.e. all edges e with y_e = 0 have their "extreme" vertices on one side of the cut)
    # (i.e. clique surrounding vertices of y_e are in the same partition)

    for i, j in nx_g.edges:
        Q[(i, j)] += nx_g.edges[i, j]['weight']
        #NOTE: may have to compute clique around nx_g.edges[i, j] to find value of y_uv?
        # leaving at implied y_uv = 1 for now, may cause issues later

    #H_QUBO_penalty
    #NOTE: may have an offset of +1?

    src = [x for x in nx_g.nodes() if x == 's']
    sink = [x for x in nx_g.nodes() if x == 't']

    Q[(src[0], src[0])] += alpha * -1
    Q[(sink[0], sink[0])] += alpha * -1
    Q[(src[0], sink[0])] += alpha * 2

    for u, v in nx_g.edges():
        Q[(u, u)] += alpha * 1   # x_u
        Q[(v, v)] += alpha * 1   # + x_v
        Q[(u, v)] += alpha * 2   # + 2w_uv
        Q[(u, v)] += alpha * 1   # + x_u*y_uv
        Q[(v, u)] += alpha * 1   # + x_v*y_uv
        Q[(u, v)] += alpha * -2  # - 2x_u * w_uv
        Q[(v, u)] += alpha * -2  # - 2x_v * w_uv
        Q[(u, v)] += alpha * -2  # - 2y_uv * w_uv

        #NOTE: format keys for y_uv/w_uv/etc like Q[(u*v, u*v)] ?

    '''


    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, chain_strength=20, num_reads=1000)

    sample = sampleset.first.sample

    print("\nProcessing solution...")
    disp_im = np.zeros((l_test_sm.shape[0], l_test_sm.shape[1]))
    print("sample shape is: ", len(sample.items()))
    #for key, val in sample.items():
    #    print("key, val = ", key, val)
    '''
    for key, val in sample.items():
        # NOTE: workaround for s & t nodes due to (str % int) error otherwise
        if key != 's' and key != 't':
            x, y = unindexing(key, l_test_sm)
            disp_im[x, y] = val
        elif key == 's':
            x, y = 0, 0
            disp_im[x, y] = val
        elif key == 't':
            x, y = l_test_sm.shape[0]-1, l_test_sm.shape[1]-1
            disp_im[x, y] = val
    '''
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(l_test_sm, cmap='Greys')
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.imshow(disp_im, cmap='Greys')
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    plt.savefig("output.png")
    '''
    print("\nOutput file generated successfully")






    #g_tiny = kolmogorov_graph(l_patch, r_patch)
    '''
    print(g_tiny.get_node_count())
    print(g_tiny.get_edge_count())
    nx_g = g_tiny.get_nx_graph()
    print(len(nx_g.nodes))
    print(len(nx_g.edges))
    '''

    '''
    disp = kolmogorov_graph(l_test_sm, r_test_sm)
    disp1 = handle_infinities_scale_color(disp)
    cv2.imwrite("test_gc_disp1.png", disp1)
    '''



    #print(l_patch)

    # set up image graphs

    #test out mf
    '''
    nodeids_l, g_l = create_graph(l_patch, 5, 5, 5)
    nodeids_r, g_r = create_graph(r_patch, 5, 5, 5)
    #plot_graph_3d(g, nodeids.shape)
    #print(g_l.get_node_count())
    #print(g_l.get_edge_count())

    nx_g_l = g_l.get_nx_graph()
    nx_g_r = g_r.get_nx_graph()
    print(len(nx_g_l.nodes))
    print(len(nx_g_l.edges))
    print(len(nx_g_r.nodes))
    print(len(nx_g_r.edges))
    '''

    #create_image_graph(l_patch, 5, 5, 5)

    #set up Q-matrix

   # Q = defaultdict(int)

    #fill in Q-matrix (hard coded for now)


    #send to DWave

    #parse cuts, assemble final disparity image

main()







