from shrink.config.strings import *
from copy import deepcopy
from datamodel import DataModel
from gen_utils import *
from itertools import combinations

import csv
import math
import matplotlib.pyplot as plt
import networkx as nx
import os
import statistics as stats


class Experimenter:
    """Execute and manage machine learning experiments"""
    def __init__(self, dm):
        self.dm = dm


    def set_datamodel(self, dm):
        self.dm = dm
        return None

    
    def get_accurate_result(self):
        return None


    def get_estimated_result(self, undergrad_grad, pgm_grad, att_dict):
        undergrad = att_dict[U_UNIVERSITY_CODE]
        grad = att_dict[UNIVERSITY]
        pgm = att_dict[PROGRAM_CODE]
        if undergrad_grad.has_edge(undergrad,grad) is False or \
                pgm_grad.has_edge(pgm,grad) is False:
                    return 0
        else:
            u_g_edges = undergrad_grad.edges()
            out_undergrad_total = \
                    sum([undergrad_grad[u][v]['weight'] \
                    for u,v in u_g_edges if u==undergrad or v==undergrad])
            out_undergrad_grad = undergrad_grad[undergrad][grad]['weight']
            frac_1 = float(out_undergrad_grad) / out_undergrad_total

            p_g_edges = pgm_grad.edges()
            out_pgm_total = \
                    sum([pgm_grad[u][v]['weight'] \
                    for u,v in p_g_edges if u==pgm or v==pgm])
            out_pgm_grad = pgm_grad[pgm][grad]['weight']
            frac_2 = float(out_pgm_grad) / out_pgm_total
            return frac_1 * frac_2


    def get_grad_uni_summary_graph(self, attr):
        G = nx.Graph()
        uni_labels = self.dm.get_uni_labels()
        for row in self.dm.data:
            u_uni = row[attr]
            if u_uni not in G:
                G.add_node(u_uni)
            for label in uni_labels:
                if label in row and row[label] == ADMIT:
                    if label not in G:
                        G.add_node(label)
                        G.add_edge(u_uni,label,weight=1)
                    else:
                        if (u_uni,label) not in G.edges():
                            G.add_edge(u_uni,label,weight=1)
                        else:
                            G[u_uni][label]['weight'] += 1
        #nx.draw_networkx(G,with_labels=True,)
        #plt.savefig('../data/results/figures/summary_graph.png')
        #plt.show()
        return G
