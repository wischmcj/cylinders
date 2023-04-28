import os 
from pathlib import Path

import calendar
import time

current_GMT = time.gmtime()

time_stamp = calendar.timegm(current_GMT)

DIRTORUN= r'C:/Users/wisch/Documents/GitProjects/projectCylinders/data/' #r'G:/My Drive/forOthers/johnVanStann/cylinders/data'
LOGDIR = r'C:/Users/wisch/Documents/GitProjects/projectCylinders/logs/log_' + str(time_stamp) + '.txt'
GRAPHLOC = r'C:/Users/wisch/Documents/GitProjects/projectCylinders/graphs/'
STEMLOC= r'C:/Users/wisch/Documents/GitProjects/projectCylinders/stems/'
# os.chdir(DIRTORUN)

# fullPath = Path(DIRTORUN)

# paths = sorted(fullPath.iterdir(),key=os.path.getmtime)

# fileNames = [f.name for f in paths if f.suffix == '.csv']

# print(fileNames)
