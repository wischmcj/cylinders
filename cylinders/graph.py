# Defining our graphs 
#   - how do you account for having multiple variant properties 
#       - e.g. is weight angle, thickness, depth of ridge, bark type?
#       - likely a combination of these - getting at the carrying capacity
#   - What do we want to calculate 
#       - Connectivity isnt that important, rather paths of least resistance 
#       - % carrying capacity full in different conditions 
#   - Starting points and volumes 
#       -number of leaves, thickenss of branches, 
#       -Technically these are everywhere that rain can land 
#   - Are leaves cylinders? or rather, circles
#   - What are the correect vectors to use ?
#       - Water doesnt often flow along the axial
#       - some trees with ridge bark warrant many around the branch 
#       - Many would have curved/ step functioned defined as they go down the tree
#       
import pickle 
import networkx as nx

class PERI(nx.Graph):
    
    #initialize our object level variables for cylider objects 
    
    def pickle_save(graph, name, path="./"):
        pickle.dump(graph, open(path + name, 'w'))
    
    def pickle_load(fullpath):
        pickle.load(open(fullpath, 'w'))

    def 