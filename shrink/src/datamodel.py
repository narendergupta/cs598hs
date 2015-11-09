from shrink.config.strings import *
from copy import deepcopy
from gen_utils import *

import csv
import random
import re


class DataModel:
    """Class for reading and managing raw data"""
    def __init__(self, data='../data/v2_norm.csv'):
        if type(data) is str and does_file_exist(data) is True:
            self.data_file = data
            self.data = []
        elif isinstance(data, DataModel) is True:
            self.data = deepcopy(data.data)
            self.uni_data_map = deepcopy(data.get_uni_data_map())


    def read_data(self, to_read_count=-1, normalize_data=True):
        self.data = []
        """Reads data file"""
        read_count = 0
        with open(self.data_file,'r') as data_f:
            reader = csv.DictReader(data_f)
            for row in reader:
                self.data.append(lowercase(row))
                read_count += 1
                if to_read_count > 0 and read_count >= to_read_count:
                    break
        # Normalize various features of datapoints
        if normalize_data is True:
            self.normalize_data()
        return None


    def write_data(self, output_file):
        with open(output_file,'w') as output_f:
            data_writer = csv.DictWriter(output_f, fieldnames=sorted(list(self.data[0].keys())))
            data_writer.writeheader()
            for row in self.data:
                data_writer.writerow(row)
        return None


    def set_data(self, data):
        self.data = data
        try:
            del self.uni_data_map
        except AttributeError:
            pass
        return None


    def filter_data(self, filter_type, feature, shortlist=None, min_v=None, max_v=None):
        if filter_type == LIST and \
                shortlist is not None and type(shortlist) is list:
                    return list(filter(
                        lambda x:x[feature] in shortlist,
                        self.data))
        elif filter_type == RANGE and min_v is not None and max_v is not None:
                    return list(filter(
                        lambda x:x[feature] != '' and \
                                float(x[feature])>=min_v and \
                                float(x[feature])<=max_v,
                        self.data))
        else:
            return self.data


    def normalize_data(self):
        norm_data = []
        for row in self.data:
            (row[TERM], row[YEAR]) = self.__normalize_term_year(lowercase(row[TERM_YEAR]))
            (row[GRE_QUANT_NORM], row[GRE_VERBAL_NORM], row[GRE_AWA_NORM]) = \
                    self.__normalize_gre_components(row[GRE_QUANT], row[GRE_VERBAL], row[GRE_AWA])
            row[TOEFL_NORM] = self.__normalize_toefl_score(row[TOEFL])
            (row[U_GRADE_NORM], row[U_GRADE_TOP_NORM], row[U_GRADE_SCALE_NORM]) = \
                    self.__normalize_ugpa(row[U_GRADE], row[U_GRADE_TOP], row[U_GRADE_SCALE])
            row[PROGRAM_CODE] = self.__normalize_program(lowercase(row[PROGRAM]))
            row[U_UNIVERSITY_CODE] = self.__get_uni_url(row[U_UNIVERSITY])[row[U_UNIVERSITY]]
            row[U_UNIVERSITY_RANK] = self.__get_undergrad_uni_rank(row[U_UNIVERSITY])
            row[MAJOR_CODE] = self.__get_major_code(row[MAJOR])[row[MAJOR]]
            row[U_MAJOR_CODE] = self.__get_major_code(row[U_MAJOR])[row[U_MAJOR]]
            uni_rank = self.__get_uni_rank(row[U_UNIVERSITY])
            for key in uni_rank.keys():
                row[key] = uni_rank[key]
            norm_data.append(row)
        self.data = norm_data
        return None


    def __normalize_term_year(self, term_year):
        term_regex = re.compile(r'fall|spring|summer')
        year_regex = re.compile(r'[\d]{4}')
        terms = term_regex.findall(term_year)
        years = year_regex.findall(term_year)
        term = terms[0] if len(terms) > 0 else None
        # Fix issues like 2109 => 2009
        if len(years) > 0:
            year_two_digit = years[0][-2:]
            if year_two_digit >= '00' and year_two_digit < '20':
                year = '20' + year_two_digit
            else:
                year = '19' + year_two_digit
        else:
            year = ''
        return (term, year)


    def __normalize_gre_components(self, quant, verbal, awa):
        quant = float(quant) if quant != '' else 0
        verbal = float(verbal) if verbal != '' else 0
        awa = float(awa) if awa != '' else 0
        if(awa<0 or awa>6):
            awa = 0.0
        if(quant>=130 and quant<=170 and int(verbal)==0):
            return (((quant-130)*100)/40, 0.0, awa)
        elif(verbal>=130 and verbal<=170 and int(quant)==0):
            return (0.0, ((verbal-130)*100)/40, awa)
        elif(quant>=200 and quant<=800 and int(verbal)==0):
            return (((quant-200)*100)/600, 0.0, awa)
        elif(verbal>=200 and verbal<=800 and int(quant)==0):
            return (0.0, ((verbal-200)*100)/600, awa)
        elif(quant>=130 and quant<=170 and verbal>=130 and verbal<=170):
            return (((quant-130)*100)/40, ((verbal-130)*100)/40, awa)
        elif(quant>=200 and quant<=800 and verbal>=200 and verbal<=800):
            return (((quant-200)*100)/600, ((verbal-200)*100)/600, awa)
        else:
            return (0.0, 0.0, awa)
    

    def __normalize_toefl_score(self, toefl):
        toefl = float(toefl) if toefl != '' else 0
        if(toefl>=0 and toefl<=120):
            toefl = (toefl*100)/120
        elif(toefl>120 and toefl<=300):
            toefl = (toefl*100)/300
        elif(toefl>=311 and toefl<=677):
            toefl = ((toefl-311)*100)/(677-311)
        else:
            toefl = 0.0
        return toefl

    
    def __normalize_ugpa(self, gpa, gpa_top, gpa_scale):
        gpa = float(gpa) if gpa != '' else 0
        gpa_top = float(gpa_top) if gpa_top != '' else 0
        gpa_scale = float(gpa_scale) if gpa_scale != '' else 0
        if int(gpa)==0 or int(gpa_scale)==0 or gpa>gpa_scale:
            gpa = 0
            gpa_top = 0
        else:
            gpa = (gpa*100)/gpa_scale
            gpa_top = (gpa_top*100)/gpa_scale
        return (gpa, gpa_top, 100)


    def __normalize_program(self, program):
        if 'phd' in program:
            return 'phd'
        else:
            return 'ms'


    def __get_major_code(self, majors):
        try:
            result = {}
            if type(majors) is str:
                majors = [majors]
            for major in majors:
                if lowercase(major) not in self.major_map.keys():
                    result[major] = ''
                else:
                    result[major] = self.major_map[lowercase(major)]
            return result
        # If self.major_map doesn't exist, create it
        except AttributeError:
            self.major_map = {}
            with open('../resources/major_codes.csv','r') as majors_f:
                major_reader = csv.DictReader(majors_f)
                for row in major_reader:
                    self.major_map[lowercase(row[MAJOR])] = row[MAJOR_CODE]
            return self.__get_major_code(majors)
        return None


    def __get_uni_url(self, unis):
        try:
            result = {}
            if type(unis) is str:
                unis = [unis]
            for uni in unis:
                if lowercase(uni) not in self.uni_urls.keys():
                    result[uni] = ''
                else:
                    result[uni] = self.uni_urls[lowercase(uni)]
            return result
        # If self.uni_urls doesn't exist, create it
        except AttributeError:
            self.uni_urls = {}
            with open('../resources/uni_urls.csv','r') as uni_urls_f:
                url_reader = csv.DictReader(uni_urls_f)
                for row in url_reader:
                    self.uni_urls[lowercase(row[UNIVERSITY])] = row[URL]
            return self.__get_uni_url(unis)
        return None


    def __get_uni_rank(self, uni_name, uni_url=None,major=None):
        if uni_url is None:
            uni_url = self.__get_uni_url(uni_name)[uni_name]
        default_rank = {SHANGHAI_MIN:0, SHANGHAI_MAX:1000, \
                QS_MIN:0, QS_MAX:1000, USNEWS:1000}
        try:
            if uni_url in self.uni_ranks.keys():
                return self.uni_ranks[uni_url]
            else:
                return default_rank
        # If self.uni_ranks doesn't exist, create it
        except AttributeError:
            self.uni_ranks = {}
            with open('../resources/uni_ranks.csv') as ranks_f:
                rank_reader = csv.DictReader(ranks_f)
                for row in rank_reader:
                    url = row[URL]
                    self.uni_ranks[url] = deepcopy(default_rank)
                    if row[USNEWS] != '':
                        self.uni_ranks[url][USNEWS] = int(row[USNEWS])
                    if row[SHANGHAI] != '':
                        parts = row[SHANGHAI].split('-')
                        if parts[0] != '':
                            self.uni_ranks[url][SHANGHAI_MIN] = int(parts[0])
                        if parts[-1] != '':
                            self.uni_ranks[url][SHANGHAI_MAX] = int(parts[-1])
                    if row[QS] != '':
                        parts = row[QS].split('-')
                        if parts[0] != '':
                            self.uni_ranks[url][QS_MIN] = int(parts[0])
                        if parts[-1] != '':
                            self.uni_ranks[url][QS_MAX] = int(parts[-1])
            return self.__get_uni_rank(uni_name, uni_url, major)
        return None


    def __get_undergrad_uni_rank(self, uni_name, uni_url=None,major=None):
        if uni_url is None:
            uni_url = self.__get_uni_url(uni_name)[uni_name]
        default_rank = ''
        try:
            if uni_url in self.u_uni_ranks.keys():
                return self.u_uni_ranks[uni_url]
            else:
                return default_rank
        # If self.uni_ranks doesn't exist, create it
        except AttributeError:
            self.u_uni_ranks = {}
            with open('../resources/undergrad_uni_ranks.csv') as ranks_f:
                rank_reader = csv.DictReader(ranks_f)
                for row in rank_reader:
                    self.u_uni_ranks[row[URL]] = row[ADHOC_RANK]
            return self.__get_undergrad_uni_rank(uni_name, uni_url, major)
        return None


    def get_uni_labels(self):
        try:
            return self.uni_labels
        # Create the uni_labels map if it's been called for the first time
        except AttributeError:
            self.uni_labels = []
            with open('../resources/uni_label_list.csv') as labels_f:
                label_reader = csv.DictReader(labels_f)
                self.uni_labels = [x[UNIVERSITY] for x in label_reader]
            return self.get_uni_labels()
        return None


    def __get_datarow_without_labels(self, row):
        labels = self.get_uni_labels()
        redacted_row = deepcopy(row)
        for label in labels:
            redacted_row.pop(label,None)
        return redacted_row


    def get_uni_data_map(self, input_unis=None):
        default_map = {ADMIT:[], REJECT:[], RESULT_NA:[]}
        try:
            if input_unis is None:
                return self.uni_data_map
            elif type(input_unis) is str:
                input_unis = [input_unis]
            result = {}
            for input_uni in input_unis:
                if input_uni in self.uni_data_map.keys():
                    result[input_uni] = self.uni_data_map[input_uni]
                else:
                    result[input_uni] = deepcopy(default_map)
            return result
        # Create the uni_data_map if it's been called for the first time
        except AttributeError:
            self.uni_data_map = {}
            labels = self.get_uni_labels()
            for label in labels:
                self.uni_data_map[label] = {ADMIT:[], REJECT:[], RESULT_NA:[]}
            for row in self.data:
                redacted_row = self.__get_datarow_without_labels(row)
                for uni in row.keys():
                    if uni not in labels:
                        continue
                    elif row[uni] == ADMIT:
                        self.uni_data_map[uni][ADMIT].append(redacted_row)
                    elif row[uni] == REJECT:
                        self.uni_data_map[uni][REJECT].append(redacted_row)
                    elif row[uni] == RESULT_NA:
                        self.uni_data_map[uni][RESULT_NA].append(redacted_row)
            return self.get_uni_data_map(input_unis)
        return None


    def save_uni_stats(self, output_file):
        uni_data_map = self.get_uni_data_map()
        with open(output_file,'w') as stats_f:
            stat_writer = csv.DictWriter(stats_f, \
                    fieldnames=[UNIVERSITY,ADMIT,REJECT,ADMIT_PLUS_REJECT,RESULT_NA])
            stat_writer.writeheader()
            for uni in uni_data_map.keys():
                admit = len(uni_data_map[uni][ADMIT])
                reject = len(uni_data_map[uni][REJECT])
                na = len(uni_data_map[uni][RESULT_NA])
                stat_writer.writerow(\
                        {UNIVERSITY:uni, ADMIT:admit, REJECT:reject, \
                        ADMIT_PLUS_REJECT: admit+reject, RESULT_NA:na})
        return None


