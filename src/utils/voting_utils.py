from collections import  Counter
from typing import List



def modified_plurality(votes:List[object])->object:
	"""
    Implementation of modified plurality when we have ties in our classifier
	"""
	data  = Counter(votes)
	most_common = data.most_common(2)
	if(len(most_common) == 2):
		if(most_common[0][1] == most_common[1][1]):
			return modified_plurality(votes[:len(votes)-1])
	return most_common[0][0]
