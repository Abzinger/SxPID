from sys import path
path.insert(0,"../sxpid")
import pickle 
#--------
# Test!
#-------

# Format of the pdf is 
# dict( (s1,s2,t) : p(s1,s2,t) ) for all s1 in S1, s2 in S2, and t in T if p(s1,s2,t) > 0.


# Read lattices from a file
# Pickled as { n -> [{alpha -> children}, (alpha_1,...) ] }
f = open("../sxpid/lattices.pkl", "rb")
lattices = pickle.load(f)
print(lattices[4][0][((1,2,3,4,),)])
print(lattices[4][0][((1,2,3,),)])
print(lattices[4][0][((1,2,4,),)])
print(lattices[4][0][((1,3,4,),)])
print(lattices[4][0][((1,2,3,),(1,2,4))])
print(lattices[4][0][((1,2,3,),(1,3,4))])
print(lattices[4][0][((1,2,4,),(1,3,4))])
print(lattices[4][0][((1,2,3,),(1,2,4,),(1,3,4))])
print(lattices[4][0][((1,2,),)])
print(lattices[4][0][((1,3,),)])
print(lattices[4][0][((1,4,),)])
print(lattices[4][0][((1,2,),(1,3,4,))])
print(lattices[4][0][((1,3,),(1,2,4,))])
print(lattices[4][0][((1,4,),(1,2,3,))])
print(lattices[4][0][((1,2,),(1,3,))])
print(lattices[4][0][((1,2,),(1,4,))])
print(lattices[4][0][((1,3,),(1,4,))])
print(lattices[4][0][((1,2,),(1,3,),(1,4))])
print("children to be omitted")
print(lattices[4][0][((2,3,4,),)])
print(lattices[4][0][((1,2,3,),(2,3,4))])
print(lattices[4][0][((1,2,4,),(2,3,4))])
print(lattices[4][0][((1,3,4,),(2,3,4))])
print(lattices[4][0][((2,3,),)])
print(lattices[4][0][((2,4,),)])
print(lattices[4][0][((3,4,),)])
print(lattices[4][0][((1,2,3,),(1,2,4,),(2,3,4))])
print(lattices[4][0][((1,2,3,),(1,3,4,),(2,3,4))])
print(lattices[4][0][((1,2,4,),(1,3,4,),(2,3,4))])
print(lattices[4][0][((1,2,),(2,3,4,))])
print(lattices[4][0][((1,3,),(2,3,4,))])
print(lattices[4][0][((1,4,),(2,3,4,))])
print(lattices[4][0][((2,3,),(1,2,4,))])
print(lattices[4][0][((2,4,),(1,2,3,))])
print(lattices[4][0][((2,3,),(1,3,4,))])
print(lattices[4][0][((2,4,),(1,3,4,))])
print(lattices[4][0][((3,4,),(1,2,4,))])
print(lattices[4][0][((3,4,),(1,2,3,))])
print(lattices[4][0][((1,2,3,),(1,2,4,),(1,3,4),(2,3,4))])
print(lattices[4][1])

