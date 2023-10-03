import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pathlib import Path
from random import random
from multiprocessing import Pool

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, transform
from descartes import PolygonPatch
from mpl_toolkits import mplot3d
from pickle import dump, load


from time import sleep

import networkx as nx
import openpyxl
import geopandas as geo
import numpy as np
import calendar
import time
import copy
import math
import logging
import settings
import os

import global_vars


from pandas import to_excel as pd

time_stamp = str(calendar.timegm(current_GMT))

NAME = "Cylinder"
logging.basicConfig(filename=''.join(['log_',str(time_stamp)])  , filemode='w', level=logging.DEBUG, encoding='utf-8',level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("my-logger")
 
#Class intented to be the workhorse that manages cylinders and the like 
class Forester:

    #initialize our object level variables for cylider objects 
    def __init__(self, filename) -> None:
        self.variable = v
        
   
    def network_simplex():
        #can be used to calculate flows on graphs with demands 
        #we could set a demand of generates X volume of flow 