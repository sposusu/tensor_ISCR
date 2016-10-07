from collections import defaultdict
from glob import glob
import operator
import os
import pickle
import sys
#############################
#     Document Model        #
#############################
def readLex(fname):
  fin = file(fname)
  lex = {}
  num = 0
  for line in fin.readlines():
    word = line.replace('\n','')
    num += 1
    lex[word] = num
  return lex

def readBackground(fname,lex):
    fin = file(fname)
    background = {}
    for line in fin.readlines():
	pair = line.replace('\n','').split()
	wordID = lex[pair[0]]
	val = float(pair[1])
	background[wordID] = val
    fin.close()
    return background

def readInvIndex(fname):
 fin = file(fname)
 inv_index = {}
 for line in fin.readlines():
	pair = line.replace('\n','').split('\t')
	wordID = int(pair[0])
	if pair[1]=='':
	    inv_index[wordID] = {}
	    continue
	docset = {}
	for valpair in pair[1].split():
	    key, val = valpair.split(':')
	    docset[int(key)] = float(val)
	inv_index[wordID] = docset
 fin.close()
 return inv_index

def readDocLength(fname):
    fin = file(fname)
    docLengs = {}
    for line in fin.readlines():
	pair = line.replace('\n','').split()
	docID = docNameToIndex(pair[0])
	docleng = float(pair[1])
	docLengs[docID] = docleng
    fin.close()
    return docLengs

def readDocModel(fname):
  fout = file(fname)
  model = {}
  for line in fout.readlines():
    tokens = line.split()
    word = int(tokens[0])
    val = float(tokens[1])
    model[word] = val
  return model

##############################
#       Simulated User       #
##############################

def readKeytermlist(keyterm_dir, query_dict):
    keyterms = defaultdict(float)

    for word_id, prob in query_dict.iteritems():
        filepath = os.path.join(keyterm_dir,str(word_id))
        with open(filepath,'r') as fin:
            for idx, line in enumerate(fin.readlines(),1):
                if idx > 100:
                    break

                pair = line.split()

                keyterms[ int(pair[0]) ] += prob * float(pair[1])

    sorted_keyterm_list = sorted(keyterms.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_keyterm_list

def readRequestlist(request_dir,fileIDs):
    requests = defaultdict(float)
    for fileID in fileIDs.keys():
        filepath = os.path.join(request_dir,str(fileID))
        with open(filepath,'r') as fin:
            for line in fin.readlines():
                pair = line.split()
                requests[ int(pair[0]) ] += float(pair[1])

    request_list = sorted(requests.iteritems(),key=operator.itemgetter(1),reverse=True)
    return request_list

def readTopicWords(topic_dir):
    topic_word_list = []

    for topic_filepath in sorted(glob(os.path.join(topic_dir,'*'))):
        if not os.path.isfile(topic_filepath):
            continue

        words = {}
        with open(topic_filepath,'r') as f:
            for line in f.readlines():
                pair = line.split()
                words[ int(pair[0]) ] = float(pair[1])
        topic_word_list.append(words)

    return topic_word_list

def readTopicList(ranking_dir,query_idx):
    ranking = []
    ranking_filepath = os.path.join(ranking_dir,str(query_idx))
    with open(ranking_filepath,'r') as fin:
        for line in fin.readlines():
            tokens = line.split()
            ranking.append((int(float(tokens[0])),float(tokens[1])))
    return ranking

def save_to_pickle(filepath,obj):
    with open(filepath,'wb') as f:
        pickle.dump(obj,f)

def load_from_pickle(filepath):
    with open(filepath,'rb') as f:
        return pickle.load(f)

def docNameToIndex(fname):
    return int(fname[1:])

def pickle_searchengine(data_dir):
    searchengine_pickle = os.path.join(data_dir,'searchengine.pickle')
    if not os.path.exists(searchengine_pickle):
        print("Pickling searchengine for {}".format(data_dir))
        lex_file = os.path.join(data_dir,'reference.lex')
        background_file = os.path.join(data_dir,'reference.background')
        inv_index_file = os.path.join(data_dir,'reference.index')
        doclength_file = os.path.join(data_dir,'reference.doclength')

        lex_dict = readLex(lex_file)
        background = readBackground(background_file,lex_dict)
        inv_index = readInvIndex(inv_index_file)
        doclength = readDocLength(doclength_file)


        obj = (lex_dict, background, inv_index, doclength )
        save_to_pickle(searchengine_pickle,obj)
    else:
        print("Search engine pickle already exists {}".format(searchengine_pickle))

def pickle_docmodels(docmodel_cache,docmodel_pickle):
    if not os.path.exists(docmodel_pickle):
        print("Pickling docmodels to {}".format(docmodel_pickle))
        docmodels, doclength_dict = load_from_pickle(docmodel_cache)
        save_to_pickle(docmodel_pickle, docmodels)
    else:
        print("Doc models pickle already exists at {}".format(docmodel_pickle))

if __name__ == "__main__":
    # Save to pickle
    data_dir = '/home/ubuntu/InteractiveRetrieval/data/reference'

    pickle_searchengine(data_dir)

    docmodel_cachepath = os.path.join(data_dir,'docmodels.cache')
    docmodel_pickle    = os.path.join(data_dir,'docmodels.pickle')

    pickle_docmodels(docmodel_cachepath, docmodel_pickle)

    #topic_dir = os.path.join(data_dir,'lda')
    #topic_pickle = os.path.join(topic_dir,'topiclist.pickle')
    #topic_list = readTopicWords(topic_dir)
    #save_to_pickle(topic_pickle,topic_list)
