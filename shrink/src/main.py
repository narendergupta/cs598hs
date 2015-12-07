from datamodel import DataModel
from experimenter import Experimenter

from shrink.config.strings import *
import argparse
import time

def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(normalize_data=False)
    dm.set_data(
            dm.filter_data(
                filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee','cs']))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR]
    attr_list = [U_UNIVERSITY_CODE, PROGRAM_CODE, UNIVERSITY]
    exp = Experimenter(dm, attr_list)
    print "Set up data complete"

    #=======PREPROCESSING========#
    '''
    summary_graphs = exp.get_all_summary_graphs(attr_list)
    print "Got all summary graphs"
    print summary_graphs.keys()
    print len(summary_graphs)
    '''
    #=======PREPROCESSING========#

    undergrad_to_grad_uni = exp.get_grad_uni_summary_graph(U_UNIVERSITY_CODE)
    pgm_code_to_ugrad_uni = exp.get_summary_graph(U_UNIVERSITY_CODE, PROGRAM_CODE)
#    print(len(undergrad_to_grad_uni.edges()), len(pgm_code_to_ugrad_uni.edges()))

#    ug_to_g, pgm_to_ug, pgm_to_g = exp.dummy_graphs()
#    print(len(ug_to_g.edges()), len(pgm_to_ug.edges()))

    #NAIVE BAYES ESTIMATION
#    att_dict = {U_UNIVERSITY_CODE:'www.bits-pilani.ac.in', \
#            PROGRAM_CODE: 'ms', \
#            UNIVERSITY: 'North Carolina State University'}
#    est = exp.get_estimated_result(undergrad_to_grad_uni,
#            pgm_code_to_ugrad_uni, att_dict)
    print "--------------------------------------------"
    #ACTUAL RESULT
    given_dict = {U_UNIVERSITY_CODE: 'www.bits-pilani.ac.in'}
    inf_dict = {PROGRAM_CODE: 'ms', UNIVERSITY: 'North Carolina State University'}
    acc = exp.get_actual_result(given_dict, inf_dict)
    print "Got actual result"
    est = exp.generic_get_estimated_result(given_dict, inf_dict)
    print "Got estimated result"
    print(est, acc)
    return None
#========TEST CONDITIONAL PROBABILITY========#
#    summary_graph_dict = {(U_UNIVERSITY_CODE, UNIVERSITY):ug_to_g, (PROGRAM_CODE, U_UNIVERSITY_CODE):pgm_to_ug, (PROGRAM_CODE, UNIVERSITY):pgm_to_g}
#    given_dict = {PROGRAM_CODE: 'ms', U_UNIVERSITY_CODE: 'www.13.com'}
#    inf_dict = {UNIVERSITY: 'Stanford'}

#    print(exp.get_total_count(ug_to_g))
#    print(exp.get_conditional_probability({(U_UNIVERSITY_CODE, UNIVERSITY):ug_to_g}, U_UNIVERSITY_CODE, UNIVERSITY, 'www.13.com', 'Stanford'))
#    print(exp.get_prior_probability(ug_to_g, 'Stanford'))
#    print(exp.generic_get_estimated_result(summary_graph_dict, given_dict, inf_dict))
#    print(exp.get_numerator(summary_graph_dict, given_dict, inf_dict))
#    print(exp.get_denominator(summary_graph_dict, given_dict))
#    return None
#========TEST CONDITIONAL PROBABILITY========#

    """
    att_dict = {U_UNIVERSITY_CODE: 'www.13.com', \
            PROGRAM_CODE: 'ms', \
            UNIVERSITY: 'Stanford'}

    est = exp.get_estimated_result(ug_to_g, pgm_to_ug, att_dict)
    print(est)
    return None
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    main(args)

