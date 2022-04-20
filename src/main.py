from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import neal
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
from graph_generator import gen_digraph_5, gen_digraph_20, gen_digraph_30


''' computes the alpha parameter for the provided graph '''
def compute_alpha(g, edge_list):
    # --> formula for alpha <--
    # Sum_over_edges [cost{u, v}] - Sum_over_nodes [cost{s, u} + cost{t, u}]

    a_term1 = 0
    a_term2 = 0
    seen_nodes = []

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

        if u != ('s' or 't') and u not in seen_nodes:
            for j in range(len(u_in)):
                if u_in[j][0] == 's':
                    #print("u_in:", u_in)
                    src_edge_ind = j

            for k in range(len(u_out)):
                if u_out[k][1] == 't':
                    #print("u_out:", u_out)
                    sink_edge_ind = k
            seen_nodes.append(u)

        if (src_edge_ind is not None) and (sink_edge_ind is not None):
            a_term2 += u_in[src_edge_ind][2]['weight'] + u_out[sink_edge_ind][2]['weight']

    # NOTE: it is possible for a_term1 and a_term2 to be equal, making alpha = 0
    # (this can happen with certain graph structures)
    # if this does happen, the final graph cut results will be incorrect
    # TODO: find a workaround for this case

    return a_term1 - a_term2

''' creates a Q matrix from the provided graph, to be fed into the annealer '''
def create_Q(nx_g, alpha):
    """
     --> precompute Q-matrix dimensions and index "landmarks" <--
     THE MATRIX: (2*E + V) x (2*E + V)
     GRAPH NODE INDICIES (the x_v's): [0, num_nodes)
     EDGE INDICIES, SET 1 (the y_uv's): [num_nodes, num_nodes+num_edges)
     EDGE INDICIES, SET 2 (the w_uv's): [num_nodes+num_edges, num_nodes+num_edges+num_edges)
    """

    Q_size = (2 * len(nx_g.edges())) + len(nx_g.nodes())
    Q = np.zeros((Q_size, Q_size))
    vert_ind_start = 0
    y_ind_start = vert_ind_start + len(nx_g.nodes())
    w_ind_start = y_ind_start + len(nx_g.edges())

    # helpful info when creating/debugging Q:
    
    #print("Q is ", Q_size, " x ", Q_size)
    #print("y start = ", y_ind_start, "w start = ", w_ind_start, "for a graph of ", len(nx_g.nodes()), " nodes and ",
    #      len(nx_g.edges()), " edges.")
    

    # IN NODE LIST: src at index len(nodes)-2, sink at len(nodes)-1
    src_ind = len(nx_g.nodes()) - 2
    sink_ind = len(nx_g.nodes()) - 1

    # convert edge & node dicts to lists for easy access
    e_lst = list(nx_g.edges(data=True))

    """
    ---> Q-MATRIX SETUP <---
    """
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

    """
     -----> H_QUBO_Penalty <-----
     first 'half': alpha*(1 - x_s - x_t + 2 * x_s * x_t)
    """

    Q[src_ind, src_ind] += -1 * alpha
    Q[sink_ind, sink_ind] += -1 * alpha
    Q[src_ind, sink_ind] += 2 * alpha

    """
    second 'half': alpha * Sum_over_edges [x_u + x_v + 2w_uv + x_u * y_uv + x_v * y_uv - 2x_u * w_uv - 2x_v * w_uv - 2y_uv * w_uv]
    """
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
        Q[x_u, x_u] += 1 * alpha
        Q[x_v, x_v] += 1 * alpha
        Q[w_uv, w_uv] += 2 * alpha
        Q[x_u, y_uv] += 1 * alpha
        Q[x_v, y_uv] += 1 * alpha
        Q[x_u, w_uv] += -2 * alpha
        Q[x_v, w_uv] += -2 * alpha
        Q[y_uv, w_uv] += -2 * alpha

    return Q, e_lst

def main():

    #create a simple networkX digraph
    nx_g, e_lst = gen_digraph_5()

    # compute alpha
    alpha = compute_alpha(nx_g, e_lst)

    #create the Q-matrix from the networkX graph
    Q, e_lst = create_Q(nx_g, alpha)    
    
    # If you want to verify the correctness of Q, examine it as a CSV:
    
    #with open("Q.csv", "w+") as file:
    #    csvWriter = csv.writer(file, delimiter=",")
    #    csvWriter.writerows(Q)

    #set up the simulated quantum annealer
    sampler = neal.SimulatedAnnealingSampler()
    chain_strength = np.trace(Q) // len(Q.diagonal())

    #run the annealer and extract the sample with the lowest energy
    sampleset = sampler.sample_qubo(Q, chain_strength=chain_strength, num_runs=11000)
    lowest = sampleset.first[0]
    keys = list(lowest.keys())
    vals = list(lowest.values())
    energy = sampleset.first[1]

    #use the results from the annealing sampler to determine which graph edges are cut
    cut_edges = []
    uncut_edges = []
    S0 = []
    S1 = []
    edge_list = list(nx_g.edges)
    node_list = list(nx_g.nodes)
    for i in range(len(nx_g.nodes), len(lowest) - len(nx_g.edges)):
        key = keys[i]
        val = vals[i]
        if val == 1:
            edge = edge_list[key-len(nx_g.nodes)]
            cut_edges.append(edge)
        else:
            edge = edge_list[key-len(nx_g.nodes)]
            uncut_edges.append(edge)

    for i in range(0, len(nx_g.nodes)):
        key = keys[i]
        val = vals[i]
        if val == 0:
            S0.append(node_list[key])
        if val == 1:
            S1.append(node_list[key])

    # Display best result
    pos = nx.shell_layout(nx_g)

    nx.draw_networkx_nodes(nx_g, pos, nodelist=nx_g.nodes, node_color='r')
    nx.draw_networkx_edges(nx_g, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
    nx.draw_networkx_edges(nx_g, pos, edgelist=uncut_edges, style='solid', width=3)
    nx.draw_networkx_labels(nx_g, pos)
    edgeLabels = nx.get_edge_attributes(nx_g, 'weight')
    nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edgeLabels)

    filename = "qubo_graphcut_results.png"
    plt.savefig(filename, bbox_inches='tight')
    print("\nYour plot is saved to {}".format(filename))

main()
