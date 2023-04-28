
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
    fileNames = [f.name for f in paths if  f.suffix == '.csv' ]
    return fileNames

def main():  
    print("test")
    parser = argparse.ArgumentParser(description='Take in run params')
    parser.add_argument('read_load', metavar='rl', type=str)
    parser.add_argument('int_auto', metavar='ia', type=str)
    args = parser.parse_args()
    
    fileNames = read_file_names()

    if args.int_auto == 'i':
        #Code for workin g with previously created objects here 
        print('interactive mode launched')
        c = cylinder.CylinderCollection('Secrest07-32_000000.csv')
        c.load_graph()
        print(len(c.graph.nodes()))
        
    if args.int_auto == 'a':
        stemlengths = {}
        for f in fileNames:
            c = cylinder.CylinderCollection(f)
            print("Processing: " + f)
            if args.read_load.lower() == 'load':
                #try:
                    print("Loading Graph for: " +  f)
                    c.load_graph_paths()
                    print(type(c.graph))
                    print(c.graph.number_of_nodes)
                # except: 
                #     print('No graph found for ' + f.replace('.csv',''))

            if args.read_load == 'read':
                c.read_csv()
                c.create_graph()

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

            stemlength_df.to_csv(STEMLOC + '.csv')

            print("test")


