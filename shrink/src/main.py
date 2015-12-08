from datamodel import DataModel
from experimenter import Experimenter

from shrink.config.strings import *
import argparse
import time

def main(args):
    dm = DataModel(args.data_file)
    #dm.read_data(to_read_count = 2000, normalize_data=False)
    dm.read_data(normalize_data=False)
    #dm.set_data(
    #        dm.filter_data(
    #            filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee','cs']))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR]
    #attr_list = [U_UNIVERSITY_CODE, PROGRAM_CODE, UNIVERSITY]
    attr_list = [MAJOR_CODE, PROGRAM_CODE, TERM]
    exp = Experimenter(dm, attr_list)
    print "Set up data complete"

    print "--------------------------------------------"
    #ACTUAL RESULT
    #given_dict = {U_UNIVERSITY_CODE: 'www.bits-pilani.ac.in'}
    #infer_dict = {PROGRAM_CODE: 'ms', UNIVERSITY: 'North Carolina State University'}
    given_dict = {TERM: 'fall'}
    infer_dict = {PROGRAM_CODE: 'ms', MAJOR_CODE: 'cs'}
    
    #exp.plot_datasize_vs_efficiency(given_dict, infer_dict, max_datasize=2000, \
    #        output_file='../results/figures/data_size_vs_time.png')
    exp.plot_datasize_vs_accuracy(given_dict, infer_dict, output_file = '../results/figures/data_size_vs_accuracy.png')
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

