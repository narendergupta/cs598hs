from shrink.config.strings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse
import time

def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(normalize_data=False)
    dm.set_data(
            dm.filter_data(
                filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee','cs']))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR, U_UNIVERSITY_RANK]
    exp = Experimenter(dm)
    undergrad_to_grad_uni = exp.get_grad_uni_summary_graph(U_UNIVERSITY_CODE)
    pgm_code_to_grad_uni = exp.get_grad_uni_summary_graph(PROGRAM_CODE)
    print(len(undergrad_to_grad_uni.edges()), len(pgm_code_to_grad_uni.edges()))

    att_dict = {U_UNIVERSITY_CODE:'www.unom.ac.in',\
            PROGRAM_CODE:'ms',\
            UNIVERSITY:'Arizona State University'}
    est = exp.get_estimated_result(undergrad_to_grad_uni,
            pgm_code_to_grad_uni, att_dict)
    print(est)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    main(args)

