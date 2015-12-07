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


class Experimenter:
    """Execute and manage machine learning experiments"""
    def __init__(self, dm, attr_list):
        self.dm = dm
        self.summary_graphs = None
        self.total_count = None
        self.attr_list = attr_list


    def set_datamodel(self, dm):
        self.dm = dm
        return None


#==========================================================================================================================#
    #========PREPROCESSING========#
    def get_all_summary_graphs(self):
        if self.summary_graphs is not None:
            return self.summary_graphs

        summary_graph_dict = {}
        for i in xrange(len(self.attr_list)):
            for j in xrange(i+1, len(self.attr_list)):
                not_uni = None
                if self.attr_list[i] == UNIVERSITY: not_uni = j
                elif self.attr_list[j] == UNIVERSITY: not_uni = i
                if not not_uni == None:
                    summary_graph_dict[(self.attr_list[i], self.attr_list[j])] = \
                            self.get_grad_uni_summary_graph(self.attr_list[not_uni])
                else:
                    summary_graph_dict[(self.attr_list[i], self.attr_list[j])] = \
                            self.get_summary_graph(self.attr_list[i], self.attr_list[j])
        self.summary_graphs = summary_graph_dict
        return summary_graph_dict
    #========PREPROCESSING========#


    #========GET THE PROBABILITIES AND GENERIC ESTIMATION FUNCTION========#
    def get_numerator(self, given_dict, infer_dict):
        numerator = 1
        for key1 in infer_dict:
            prod = 1
            for key2 in given_dict:
                prod *= self.get_conditional_probability(\
                        key2, key1, given_dict[key2], infer_dict[key1])
                if(prod == 0.0):
                    return 0.0
            graph_for_given_key = self.get_particular_graph(key1)
            prod = prod * self.get_prior_probability(graph_for_given_key, infer_dict[key1])
            numerator = numerator * prod
        return numerator


    def get_particular_graph(self, key):
        summary_graph_dict = self.get_all_summary_graphs()
        for attr_tup in summary_graph_dict:
            if key in attr_tup:
                if key == UNIVERSITY:
                    return summary_graph_dict[attr_tup]
                if UNIVERSITY not in attr_tup:
                    return summary_graph_dict[attr_tup]
        return None


    def get_total_count(self, graph):
        return sum([graph[u][v]['weight'] for u,v in graph.edges()])


    def get_prior_probability(self, graph, att_val):
        total_att = 0
        total_att = sum([graph[u][v]['weight'] for u,v in graph.edges() \
                if u==att_val or v == att_val])
        print("P(%s) = %f" % (att_val, float(total_att)/self.get_total_count(graph)))
        return float(total_att)/self.get_total_count(graph)


    def get_conditional_probability(self, infer_type, given_type, infer_val, given_val):
        """
        get probability of infer|given
        """
        summary_graph_dict = self.get_all_summary_graphs()
        infer_first = False
        if (infer_type, given_type) in summary_graph_dict:
            graph = summary_graph_dict[(infer_type, given_type)]
            infer_first = True
        elif (given_type, infer_type) in summary_graph_dict:
            graph = summary_graph_dict[(given_type, infer_type)]
        else:
            print("Summary graph not found! Fatal!")
            quit()
        infer = 0.0
        total = 0.0
        if infer_first:
            infer = graph.get_edge_data(infer_val, given_val)
            if not infer:
                return 0.0
            total = sum([graph[u][v]['weight'] for u,v in graph.edges() if v==given_val])
        else:
            infer = graph.get_edge_data(given_val, infer_val)
            if not infer:
                return 0.0
            total = sum([graph[u][v]['weight'] for u,v in graph.edges() if u==given_val])

        print("P(%s/%s) = %f" %(infer_val, given_val, float(infer)/total))
        return float(infer)/total


    def get_denominator(self, given_dict):
        summary_graph_dict = self.get_all_summary_graphs()
        prod = 1
        for key in given_dict:
            graph_for_given_key = self.get_particular_graph(key)
            prior_for_given_key = self.get_prior_probability(graph_for_given_key, given_dict[key])
            if(prior_for_given_key == 0):
                return -1
            prod = prod * prior_for_given_key
        return prod


    def generic_get_estimated_result(self, given_dict, infer_dict):
        numer = self.get_numerator(given_dict, infer_dict)
        denom = self.get_denominator(given_dict)**len(infer_dict)
        #Done to avoid division by zero error.
        if(denom == -1):
            return 0.0
        return numer/denom

    #========GET THE PROBABILITIES AND GENERIC ESTIMATION FUNCTION========#

#==========================================================================================================================#

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
                        passed_infer = passed_infer and \
                                (infer_dict[key] in row and row[infer_dict[key]] == ADMIT)
                    else:
                        passed_infer = passed_infer and \
                                (key in row and row[key] == infer_dict[key])
                    if not passed_infer: break
                if passed_infer: inferred_count += 1

        print inferred_count, given_count
        if(given_count == 0):
            return 0.0
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
#        plt.savefig('../results/figures/summary_graph_' + attr1_type + '_' + attr2_type + '.png')
#        plt.show()
        return G


    def get_grad_uni_summary_graph(self, attr):
        G = nx.Graph()
        uni_labels = self.dm.get_uni_labels()
        for row in self.dm.data:
            other_attr = row[attr]
            if other_attr not in G:
                G.add_node(other_attr)
            for label in uni_labels:
                if label in row and row[label] == ADMIT:
                    if label not in G:
                        G.add_node(label)
                        G.add_edge(other_attr,label,weight=1)
                    else:
                        if (other_attr,label) not in G.edges():
                            G.add_edge(other_attr,label,weight=1)
                        else:
                            G[other_attr][label]['weight'] += 1
#        nx.draw_networkx(G,with_labels=True,)
#        plt.savefig('../results/figures/summary_graph_UNIVERSITY_' + attr + '.png')
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
