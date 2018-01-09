import hashlib
import os

from collections import OrderedDict

def big5trait(wordList, traits, path):
	ret = {}
	for key, value in wordList.items():
		ret[key] = []
		tmplist = []
		for word in value:
			w_val = 1.0
			hashFile = open(os.path.join(path, "HashDict/%s" % hashlib.md5(word.encode('utf-8')).hexdigest()[:2]), "r")
			for line in hashFile:
				toks = line.split()
				if(toks.pop(0) == word):
					if(w_val >= 1):
						w_val = 0.0
					for i in range(0, 5):
						tf = float(toks[i]) - traits[i]
						w_val += tf * tf
			tmplist.append((word, w_val))
		tmplist = sorted(tmplist, key=lambda x : x[1])
		for t_word in tmplist:
			ret[key].append(t_word[0])
	return ret
