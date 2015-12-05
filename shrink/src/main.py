from datamodel import DataModel
from experimenter import Experimenter

from shrink.config.strings import *
import argparse
import time

def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=2, normalize_data=False)
    dm.set_data(
            dm.filter_data(
                filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee','cs']))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR, U_UNIVERSITY_RANK]
    exp = Experimenter(dm)

    #=======PREPROCESSING========#
    attr_list = [U_UNIVERSITY_CODE, PROGRAM_CODE, UNIVERSITY]
    summary_graphs = exp.get_all_summary_graphs(attr_list)
    print len(summary_graphs)
    #=======PREPROCESSING========#

    undergrad_to_grad_uni = exp.get_grad_uni_summary_graph(U_UNIVERSITY_CODE)
    pgm_code_to_ugrad_uni = exp.get_summary_graph(U_UNIVERSITY_CODE, PROGRAM_CODE)
    print(len(undergrad_to_grad_uni.edges()), len(pgm_code_to_ugrad_uni.edges()))
    
#    ug_to_g, pgm_to_ug, pgm_to_g = exp.dummy_graphs()
#    print(len(ug_to_g.edges()), len(pgm_to_ug.edges()))

    #NAIVE BAYES ESTIMATION 
    att_dict = {U_UNIVERSITY_CODE:'www.bits-pilani.ac.in', \
            PROGRAM_CODE: 'ms', \
            UNIVERSITY: 'North Carolina State University'}
    est = exp.get_estimated_result(undergrad_to_grad_uni,
            pgm_code_to_ugrad_uni, att_dict)
    
    #ACTUAL RESULT
    given_dict = {U_UNIVERSITY_CODE: 'www.bits-pilani.ac.in'}
    inf_dict = {PROGRAM_CODE: 'ms', UNIVERSITY: 'North Carolina State University'}
    acc = exp.get_actual_result(given_dict, inf_dict)
    
    print(est, acc)
    return None

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

