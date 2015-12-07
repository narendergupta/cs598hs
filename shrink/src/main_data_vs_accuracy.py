from datamodel import DataModel
from experimenter import Experimenter
import matplotlib.pyplot as plt
from shrink.config.strings import *
import argparse
import time

def test_data_size_vs_diff(dm, given_dict, infer_dict):
    #Read all data from data model
    dm.read_data(normalize_data=False)   
    #attr_list = [U_UNIVERSITY_CODE, PROGRAM_CODE, MAJOR_CODE]#UNIVERSITY]
    attr_list = [MAJOR_CODE, PROGRAM_CODE, TERM]
    
    #Size of data
    data_size = len(dm.data)

    #Step size = 10 steps 
    step_size = data_size//10

    #Get experiment data in a dict
    size = []
    accuracy = []

    for i in xrange(step_size, data_size, step_size):
        dm_test = DataModel("")
        dm_test.set_data(dm.data[:i])
        exp_test = Experimenter(dm_test, attr_list)
        actual = exp_test.get_actual_result(given_dict, infer_dict)
        estimation = exp_test.generic_get_estimated_result(given_dict, infer_dict)
        size.append(i)
        accuracy.append(abs(estimation - actual))
        print("Step:%d--->Actual:%f--->Estimate:%f" %(i, actual, estimation))
        print "-------------------------------------------------------------"
    plt.figure()
    plt.plot(size, accuracy)
    plt.title("Data Size vs Accuracy")
    plt.show()
   
def main(args):
    dm = DataModel(args.data_file)

    """
    dm.read_data(normalize_data=False)
    dm.set_data(
            dm.filter_data(
                filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee','cs']))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR]
    attr_list = [U_UNIVERSITY_CODE, PROGRAM_CODE, UNIVERSITY]
    exp = Experimenter(dm, attr_list)
    print "Set up data complete"
    
    undergrad_to_grad_uni = exp.get_grad_uni_summary_graph(U_UNIVERSITY_CODE)
    pgm_code_to_ugrad_uni = exp.get_summary_graph(U_UNIVERSITY_CODE, PROGRAM_CODE)
#    print(len(undergrad_to_grad_uni.edges()), len(pgm_code_to_ugrad_uni.edges()))

#    ug_to_g, pgm_to_ug, pgm_to_g = exp.dummy_graphs()
#    print(len(ug_to_g.edges()), len(pgm_to_ug.edges()))
    """
    print "--------------------------------------------"
    #ACTUAL RESULT
    #given_dict = {U_UNIVERSITY_CODE: 'www.bits-pilani.ac.in'}
    #inf_dict = {PROGRAM_CODE: 'ms', MAJOR_CODE: 'cs'}#UNIVERSITY: 'North Carolina State University'}
    given_dict = {TERM: 'fall'}
    inf_dict = {PROGRAM_CODE: 'ms', MAJOR_CODE: 'cs'}
    """
    acc = exp.get_actual_result(given_dict, inf_dict)
    print "Got actual result"
    est = exp.generic_get_estimated_result(given_dict, inf_dict)
    print "Got estimated result"
    print(est, acc)
#    print(est)
    """
    
#=======TEST DATA SIZE VS ACCURACY ========#
    test_data_size_vs_diff(dm, given_dict, inf_dict)
#=======TEST DATA SIZE VS ACCURACY ========#
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    main(args)

