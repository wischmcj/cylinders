import os 
from pathlib import Path

DIRTORUN= r'C:/Users/wisch/Documents/GitProjects/projectCylinders/data' #r'G:/My Drive/forOthers/johnVanStann/cylinders/data'

os.chdir(DIRTORUN)

fullPath = Path(DIRTORUN)

paths = sorted(fullPath.iterdir(),key=os.path.getmtime)

fileNames = [f.name for f in paths if f.suffix == '.csv']

print(fileNames)
