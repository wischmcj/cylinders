import os 
from pathlib import Path

import calendar
import time

current_GMT = time.gmtime()

time_stamp = str(calendar.timegm(current_GMT))

DIR= r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/' #r'G:/My Drive/forOthers/johnVanStann/cylinders/data'
LOGDIR = r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/output/logs/log_' + str(time_stamp) + '.txt'
GRAPHLOC = r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/output/graphs/'
STEMLOC= r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/output/stems/'
PLOC= r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/output/project/'
PLOTLOC = r'C:/Users/wisch/Documents/GitProjects/cylinders_personal/output/plots/'
# os.chdir(DIRTORUN)

# fullPath = Path(DIRTORUN)

# paths = sorted(fullPath.iterdir(),key=os.path.getmtime)

# fileNames = [f.name for f in paths if f.suffix == '.csv']

# print(fileNames)
