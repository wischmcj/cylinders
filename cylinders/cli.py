
import argparse
import geopandas as geo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from time import sleep
from random import random
from multiprocessing import Pool
from shapely.geometry import Polygon
from shapely.ops import unary_union
from descartes import PolygonPatch
import networkx as nx
from mpl_toolkits import mplot3d

from cylinders.settings import DIRTORUN, STEMLOC, GRAPHLOC
from . import base as cylinder

def read_file_names():
    os.chdir(DIRTORUN)
    fullPath = Path(DIRTORUN)
    paths = sorted(fullPath.iterdir(),key=os.path.getmtime)
    fileNames = [f.name for f in paths if f.suffix == '.csv']
    return fileNames

def main():  

    print("test")
    parser = argparse.ArgumentParser(description='Take in run params')
    parser.add_argument('read_load', metavar='N', type=str)
    parser.add_argument('cylinders', metavar='N', type=str)
    args = parser.parse_args()
    
    fileNames = read_file_names()

    stemlengths = {}
    for f in fileNames:
        if args.read_load.lower() == 'load':
            g = cylinder.pickle_load(GRAPHLOC + f)

        if args.read_load == 'read':
                print("Processing: " + f)
                c = cylinder.CylinderCollection(f)
                c.read_csv()
                c.compute_stem_paths()
                stemlengths[f] = c.stempaths
            
            savStemlengths = stemlengths

            for s in stemlengths:
                tmp = stemlengths[s].flatten()
                stemlengths[s] = tmp[~np.isnan(tmp)]

            
            stemlength_df = pd.DataFrame()
            for s in stemlengths:
                tmp = pd.DataFrame(stemlengths[s], columns = [s])
                stemlength_df = pd.concat([stemlength_df,tmp], ignore_index=True, axis=1)

            
            stemlength_df.columns = list(stemlengths.keys()) 

            hist = stemlength_df.hist(bins=100)
            
            stemlength_df.mean(axis=0)

            stemlength_df.std(axis=0)

            stemlength_df.columns = list(stemlengths.keys()) 
            
            stemlength_df.head()

            stemlength_df.to_csv(STEMLOC + 'stemlengths.csv')

            print("test")


