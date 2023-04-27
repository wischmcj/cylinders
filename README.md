<h1>CanopyHydrodymaics</h1>


<h2>Contents:</h2>

- "_src" Directories
  1. **python_src:** Refactors the code originally developed in matlab in python. Allows for the use of more robust (not to mention, free) libraries for network analysis (networkx)
  2. **matlab_src:** Original code, as of 4/27 has the additional benefit of also plotting the cyliders 

- Data 
  1. **:** contains sample files to allow for running of the program. 
  2. **Cyl Fitting Types:**  EXTRAPOLATETODTM, ALLOMETRICGROWTHVOLUME, SPHEREFOLLOWING, MEDIAN



<h3>Approach</h3>

- Pixel data from Lidar is read in to an object called cylinder
    - An array variable of said cylider (pSV) is the created that stores the projected vetors for each cylinder
    - The proejcted vecors (in conjunctuon with the parent-child relationships from the raw data) are then used to create an adjacency matrics 
    - The adjacency matrix is then converted into a graph with networkx