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

    
    def get_estimated_result(self, undergrad_grad, pgm_ugrad, att_dict):
        undergrad = att_dict[U_UNIVERSITY_CODE]
        grad = att_dict[UNIVERSITY]
        pgm = att_dict[PROGRAM_CODE]
        if undergrad_grad.has_edge(undergrad,grad) is False or \
                pgm_ugrad.has_edge(pgm,undergrad) is False:
                    return 0
        else:
            u_g_edges = undergrad_grad.edges()
            out_undergrad_total = \
                    sum([undergrad_grad[u][v]['weight'] \
                    for u,v in u_g_edges if u==undergrad or v==undergrad])
            out_undergrad_grad = undergrad_grad[undergrad][grad]['weight']
            frac_1 = float(out_undergrad_grad) / out_undergrad_total
            print "%d -- out_ug_total" % out_undergrad_total
            p_ug_edges = pgm_ugrad.edges()
            out_pgm_total = \
                    sum([pgm_ugrad[u][v]['weight'] \
                    for u,v in p_ug_edges if u==undergrad or v==undergrad])
            out_pgm_ugrad = pgm_ugrad[pgm][undergrad]['weight']
            print "%d -- out_pgm_total" % out_pgm_total
            frac_2 = float(out_pgm_ugrad) / out_pgm_total
            return frac_1 * frac_2

   
    def get_actual_result(self, given_dict, infer_dict):
        given_count = 0
        inferred_count = 0
        for row in self.dm.data:
            passed = True
            for key in given_dict:
                if key == UNIVERSITY:
                    passed = passed and (given_dict[key] in row and row[given_dict[key]] == ADMIT) 
                else:
                    passed = passed and (key in row and row[key] == given_dict[key])
                if not passed: break
            if passed:
                given_count += 1 
                passed_infer = True
                for key in infer_dict:
                    if key == UNIVERSITY:
                        passed_infer = passed_infer and (infer_dict[key] in row and row[infer_dict[key]] == ADMIT)     
                    else:
                        passed_infer = passed_infer and (key in row and row[key] == infer_dict[key])
                    if not passed_infer: break
                if passed_infer: inferred_count += 1
           
        print inferred_count, given_count 

        return float(inferred_count) / given_count 


    def get_summary_graph(self, attr1_type, attr2_type):
        G = nx.Graph()
        #uni_labels = self.dm.get_uni_labels()
        for row in self.dm.data:
            node1 = row[attr1_type]
            if node1 not in G:
                G.add_node(node1)
            node2 = row[attr2_type]
            if node2 not in G:
                G.add_node(node2)
                G.add_edge(node1,node2,weight=1)
            else:
                if (node1, node2) not in G.edges():
                    G.add_edge(node1,node2,weight=1)
                else:
                    G[node1][node2]['weight'] += 1

#        nx.draw_networkx(G,with_labels=True,)
        #plt.savefig('../data/results/figures/summary_graph.png')
 #       plt.show()
        return G

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

#        nx.draw_networkx(G,with_labels=True,)
        #plt.savefig('../data/results/figures/summary_graph.png')
#        plt.show()
        return G

#============================================================
#============================================================

    def dummy_graphs(self):
        #Graph of Undergrad to Grad
        g1 = nx.Graph()
        g1.add_node('www.13.com')
        g1.add_node('www.14.com')
        g1.add_node('Stanford')
        g1.add_node('CMU')
        g1.add_node('UIUC')
        g1.add_edge('www.13.com', 'Stanford', weight=32)
        g1.add_edge('www.13.com', 'CMU', weight=15)
        g1.add_edge('www.13.com', 'UIUC', weight=3)
        g1.add_edge('www.14.com', 'Stanford', weight=10)
        g1.add_edge('www.14.com', 'CMU', weight=8)
        g1.add_edge('www.14.com', 'UIUC', weight=7)

        #Graph of undergrad to program
        g2 = nx.Graph()
        g2.add_node('ms')
        g2.add_node('phd')
        g2.add_node('www.13.com')
        g2.add_node('www.14.com')
        g2.add_edge('www.13.com', 'ms', weight=43)
        g2.add_edge('www.13.com', 'phd', weight=7)
        g2.add_edge('www.14.com', 'ms', weight=20)
        g2.add_edge('www.14.com', 'phd', weight=5)

        #Graph of program to grad
        g3 = nx.Graph()
        g3.add_node('Stanford')
        g3.add_node('CMU')
        g3.add_node('UIUC')
        g3.add_node('ms')
        g3.add_node('phd')
        g3.add_edge('ms', 'Stanford', weight=37)
        g3.add_edge('ms', 'CMU', weight=20)
        g3.add_edge('ms', 'UIUC', weight=6)
        g3.add_edge('phd', 'Stanford', weight=5)
        g3.add_edge('phd', 'CMU', weight=3)
        g3.add_edge('phd', 'UIUC', weight=4)

        return g1, g2, g3

#============================================================
#============================================================
