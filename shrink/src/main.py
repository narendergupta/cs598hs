from shrink.config.strings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=2000,normalize_data=False)
    dm.set_data(
            dm.filter_data(
                filter_type=LIST, feature=U_MAJOR_CODE, shortlist=['ee']))
    print(len(dm.data))
    attributes_all = [U_GRADE_NORM, GRE_QUANT_NORM, GRE_VERBAL_NORM, GRE_AWA_NORM,\
            TOEFL_NORM, PROGRAM_CODE, U_MAJOR_CODE, TERM, YEAR, U_UNIVERSITY_RANK]
    exp = Experimenter(dm)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    main(args)

