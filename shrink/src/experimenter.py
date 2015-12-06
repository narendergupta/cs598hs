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
#==========================================================================================================================#
    #========PREPROCESSING========#
    def get_all_summary_graphs(self, attr_list):
        summary_graph_dict = {}

        for i in xrange(len(attr_list)):
            for j in xrange(i+1, len(attr_list)):
                not_uni = None
                if attr_list[i] == UNIVERSITY: not_uni = j
                elif attr_list[j] == UNIVERSITY: not_uni = i
                if not not_uni == None:
                    summary_graph_dict[(attr_list[i], attr_list[j])] = self.get_grad_uni_summary_graph(attr_list[not_uni])
                else:
                    summary_graph_dict[(attr_list[i], attr_list[j])] = self.get_summary_graph(attr_list[i], attr_list[j])
        return summary_graph_dict
    #========PREPROCESSING========#

    #========GET THE PROBABILITIES AND GENERIC ESTIMATION FUNCTION========#
    def get_numerator(self, summary_graph_dict, given_dict, infer_dict):
        numerator = 1
        for key1 in infer_dict:
            prod = 1
            for key2 in given_dict:
                prod = prod * self.get_conditional_probability(summary_graph_dict, key2, key1, given_dict[key2], infer_dict[key1])
            graph_for_given_key = self.get_particular_graph(summary_graph_dict, key1)
            prod = prod * self.get_prior_probability(graph_for_given_key, infer_dict[key1])
            numerator = numerator * prod

        return numerator

    def get_particular_graph(self, summary_graph_dict, key):
        for attr_tup in summary_graph_dict:
            if key in attr_tup:
                return summary_graph_dict[attr_tup]
        return None

    def get_total_count(self, graph):
        return sum([graph[u][v]['weight'] for u,v in graph.edges()])

    def get_prior_probability(self, graph, att_val):
        total_att = 0
        total_att = sum([graph[u][v]['weight'] for u,v in graph.edges() if u==att_val or v == att_val])
        print("P(%s) = %f" % (att_val, float(total_att)/self.get_total_count(graph )))
        return float(total_att)/self.get_total_count(graph)

    def get_conditional_probability(self, summary_graph_dict, infer_type, given_type, infer_val, given_val):
        """
        get probability of infer/given
        """
        infer_first = False
        if (infer_type, given_type) in summary_graph_dict:
            graph = summary_graph_dict[(infer_type, given_type)]
            infer_first = True
        elif (given_type, infer_type) in summary_graph_dict: graph = summary_graph_dict[(given_type, infer_type)]
        else: return 0
        total = 0
        if infer_first:
            total = sum([graph[u][v]['weight'] for u,v in graph.edges() if v==given_val])
        else:
            total = sum([graph[u][v]['weight'] for u,v in graph.edges() if u==given_val])
        infer = 0
        if infer_first:
            infer = graph[infer_val][given_val]['weight']
        else:
            infer = graph[given_val][infer_val]['weight']
        
        print("P(%s/%s) = %f" %(infer_val, given_val, float(infer)/total))
        return float(infer)/total

    #TODO
    '''NOTE:
      Out method is giving undue importance to one attribute when calculating the denominator.
      In fact, the way we are solving this, the current implementation re-calculates things
      many times which eventually cancel eachother out.

      P(a,b,c....z | 1,2,3....10) will simplify to:
      P(a|_phi_) * P(b|_phi_) * ....* P(z|_phi_)

      Where _phi_ can be any attribute out of the given attributes.
      In the below function, _phi_ is taken to be attr_list[0].
      There is something wrong with this since everything boils down to being
      conditioned on just one attribute out of all given attributes.

      Also, we are using Naive Bayes twice in our formula (Page 3 midterm report). This might be wrong.
    '''
    def get_denominator(self, summary_graph_dict, given_dict):
        prod = 1
        for key in given_dict:
            graph_for_given_key = self.get_particular_graph(summary_graph_dict, key)
            prior_for_given_key = self.get_prior_probability(graph_for_given_key, given_dict[key])

            prod = prod * prior_for_given_key
        return prod 
       
        """ 
        attr_list = given_dict.keys()
        given_key = attr_list[0]
        graph_for_given_key, attr_ind = self.get_particular_graph(summary_graph_dict, given_key)
        prior_for_given_key = self.get_prior_probability(graph_for_given_key, attr_ind, given_dict[given_key])
        prod = 1
        for i in xrange(1, len(attr_list)):
            prod = prod * self.get_conditional_probability(summary_graph_dict, attr_list[i], given_key, given_dict[attr_list[i]], given_dict[given_key])
        prod = prod * prior_for_given_key
        return prod
        """

    def generic_get_estimated_result(self, summary_graph_dict, given_dict, infer_dict):
        return self.get_numerator(summary_graph_dict, given_dict, infer_dict) / (self.get_denominator(summary_graph_dict, given_dict)**len(infer_dict)) 
    
    #========GET THE PROBABILITIES AND GENERIC ESTIMATION FUNCTION========#

#==========================================================================================================================#

    """
    NEED TO COME FROM GRAPH AND NOT FROM THE ENTIRE DATA

    def get_denominator_estimation(self, att_dict, num_infer):
        total_data = len(self.dm.data)
        infer_data = 0
        for row in self.dm.data:
            passed = True
            for key in att_dict:
                passed = passed and (key in row and row[key] == att_dict[key])
                if not passed: break
            if passed: infer_data += 1

        return (float(infer_data)/total_data)**num_infer
    """

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
