from shrink.config.strings import *
from copy import deepcopy
from datamodel import DataModel
from gen_utils import *
from itertools import combinations

import csv
import math
import os
import statistics as stats


class Experimenter:
    """Execute and manage machine learning experiments"""
    def __init__(self, dm):
        self.dm = dm


    def set_datamodel(self, dm):
        self.dm = dm
        return None


