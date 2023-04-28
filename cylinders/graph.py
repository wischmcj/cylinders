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
# stem flow and leaves, leaves dont seem to contribute to stem floew. 
# leaves mostly contribute to root  thhrought gloe 
# wer want to figure out how mich out current estimate of stem flow is off 
#curerently we assume a constant factor for all sitruations 
# Isotope testing idea, where is the water flowing within the ecowatershed
#   Perhaps measuring with underdueterated water 
#What about the water that is being capture within 
#look into  into voltaire 
# Nizxhea 
# for our trees all of our trees do not havemuch micro relief 
#      not alot of wierd topology 
#Most of our trees are like grey barked beech trees 
# older bark does get abit gnarly 
# We want to keep it simple, just cylinders 
# Essentially a pipe under the branch carrying water with minimal interaction 
# We have stem flow responses from rain fall to out flow down to trunk
#   basically a collar around the tree(gutter) that feeds water nto a bucket 
#      seeiing more flow than we would have expected 
# Urban trees - lack of competition, they are able to drain a much larger 
# amount fof water 
#   most of the trees Ar like birch maples, etc that do not get downward branches 
# Farmers are on board with stopping phosphorous, actually have gotten runoff very low 
#   We are still getting algall blooms, but why 
#    stem flow hs a ton more nutrients compared to other 
#   Trees are huge nutrient leechers 
#   
# Arboretum trees the we have chosen are phenotypically different that we expect to have
#
# HE is helping dig up data and research from 1800s germany and forward 
# has been finding stem frlow research from back the 
#   tthis has been ongoing as stem flow can contribute to many weird findings 
#       An unacconted for factor in hudrologica
#   We still dont have a good idea of what the key factor behind stem flow is 
#       Current models are way over complicated (opinion) 
#           Rouchbark trees v smmoothflow trees - methods of measurements has confounde
#               many experoements 
# Main question that we want to answer, what percentage, and which stemflows 
# actually make it to the trunk
#
# What about wind pushing the rain around the runk
#   the angle of the rain effecting the angle of projection for our greaph
#
# Does our model predict the data 
#   Looking at the modle in the way that hydrlogists do and seeing if itm atchs the data 
#   The hypithesis is confirmed by previosu work that the data does not match this mode l
#
# ACTUAL WORK **
#   Now that we have the stem paths, we want to:
#   Look at the individual stem paths and evalueate 
#       graphing elevation v lenght to the base and see where there might be water loss throughdrip
#   Perhaps a heat make of where the water falling under the canopy
#   What structures emerge (where do multiple branches meet, how do they connect (in a V, U etc),...)
#   Measurements show that throghfall is 60 - 70% whereas stemflow is more like 30--40%
#       the methods used to measure thi take a sample and often miss large dripoints 
#       large amount of uncertinty 
#       Look up the work of the Zimmermans 
#   Our model could fuel where thes measurements are taken in the field 
#   Application of water sheds to trees 
#       Current water shed models look at just the outflowand see if 
#   Main Goals 
#       Itendify key structural features 
#       Predict key variables effecting stem flow
#       Choose a good representiive example to represent the keu features 
#       
#  
#      
#
# What are the Concecuences
#   Help inform field experienments/measurements 
#   Can tree watershed improvements ve used to model actual water sheds 
#   Two papers 
#       Modleing and using the cyliders to measure stem flow 
#   Empower students e.g. me to write first offer papers 
#       Nick lewis is the graduate student working on the 12 trees paper 
#           He is looking at the bulk traits that we can get from the data - avg angle 
#           he is applying hydrological models in the ground 
#
#   

# Bruno l'tour
#   studies scietists to understand what is good data?
#       what is noise what is not
# Look up his joy of science translation  - joy of nietzea 
#    youtube 
#   Also alexader dumaz audio historical fiction - archive .org
#   
#Other papers - making an alterative to computree that can model straangler figs 
#   alot of interestin in strangler figs 
#   The research centers that work in these tropical arease are losing funding 
#   The moore foundation is intersted in pumping dollars into that research
#   NSF has already put alot of money onto monteverde 
#   Strangler figs have alot of implecations into the greater ecology of the area 
#   Revisions of the current QSMs for these trees can get alot of tracktion 
#   Upcomming proposal
#       How can we model the missing cloud water in the forest from the 
#   
#Potential nodes for geochemical cycling 
#   we are finding that the majority of the cycling ofmineral is haopening is nodes 
#   local places have a large amount of activity 
#
#Koalas drinking stem flow as a main source of water 
#
#NSF graduate research fellowship 
#   Can olny apply before your masters program 
#   Maybe 30K-40k a year 
#Macrosystems grant 
#   Projects that would allow us to give a product for data are key for ffunding 
#   
#   Berkley labs doing the fig work 
#




import pickle 
import networkx as nx

class PERI(nx.Graph):
    
    #initialize our object level variables for cylider objects 
    
    def pickle_save(graph, name, path="./"):
        pickle.dump(graph, open(path + name, 'w'))
    
    def pickle_load(fullpath):
        pickle.load(open(fullpath, 'w'))

    