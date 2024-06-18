import os
import numpy as np
from utils.CornerPDB import State


def swap(array,pos1,pos2):
	array[pos1], array[pos2] = array[pos2], array[pos1]
	return array

def unrank(hash,dual,distinctSize=8,puzzle_size=8):
	
	for i in range(puzzle_size):
		dual[i]=i
	
	for i in range(distinctSize):
		dual=swap(dual,i,int(i+hash%(puzzle_size-i)))
		hash = hash/(puzzle_size-i)
	
	return dual



def get_state(hash,corner_size=8):
	state=State()
	factorial=40320
	dual=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

	hash_val=hash
	hash=hash//factorial
	hash_val=hash_val%factorial
	
	dual=unrank(hash_val,dual)

	# set location of cubes
	for i in range(8):
		state.loc[dual[i]]=i
	
	# set orientation of cubes
	cnt=0
	limit=min(corner_size,7)
	for i in range(limit-1, -1, -1) :
		state.orientation[i]=hash%3
		cnt+=hash%3
		hash=hash//3 
	if corner_size==8:
		state.orientation[-1]=(3-(cnt%3))%3
	
	return state.get_nn_input()
    	


def readPDB(name):
	
	if not os.path.exists('pdbs/'+name+"/preprocessed.npy"):	
		
		dataset = []
		dist={}
		avg=0
		filename="pdbs/"+name+"/"+name+".pdb"
		headers=268
		with open(filename, "rb") as f:
			
			# read headers first
			while headers>0:
				byte=f.read(1)
				headers-=1	
			
			byte=f.read(1)	
			depth = int.from_bytes(byte, "big")
			index=0
			while byte:				
				first_part=depth//16
				second_part=depth%16 
				dataset.append([first_part,index])
				dataset.append([second_part,index+1])
				byte=f.read(1)
				depth = int.from_bytes(byte, "big")
				index+=2

				# add to the dist
				if first_part not in dist:
					dist[first_part]=1
				else:
					dist[first_part]+=1
				if second_part not in dist:
					dist[second_part]=1
				else:
					dist[second_part]+=1
				
				# compute average heuristic
				avg=avg+first_part+second_part
		
		# save the dataset as a numpy array
		np_dataset=np.array(dataset)
		np.save('pdbs/'+name+"/preprocessed.npy", np_dataset)
		
		with open("pdbs/"+name+"/"+'info.txt', 'a') as file:
			file.write("*"*50+"\n")
			file.write("average heuristic: "+str(avg/len(dataset))+"\n")
			file.write("number of entries: "+str(len(dataset))+"\n")
			file.write("heuristic distribution: \n")
			dist=dict(sorted(dist.items()))
			for key in dist: 
				file.write("value: "+str(key)+"        number:  "+str(dist[key])+"\n")
		
	
	# open the dataset
	dataset=np.load('pdbs/'+name+"/preprocessed.npy")
	np.random.shuffle(dataset)

	return dataset
	