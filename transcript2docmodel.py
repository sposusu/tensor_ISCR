from collections import defaultdict
import pickle
from glob import glob
import os

import jieba
from tqdm import tqdm

def run_transcript2docmodel(lex_file, transcript_dir, docmodels_cache, save_docmodel_dir, doclength_file, index_file):

    # Load lex dict
    lex_dict = {}
    with open(lex_file,'r') as fin:
        for idx, line in enumerate(fin.readlines(),1):
            lex = line.strip()
            lex_dict[ lex ] = idx

    # All doc lengths in one file
    if not os.path.exists(docmodels_cache):
        print("Reading docmodels from {} and save cache to {}".format(transcript_dir, docmodels_cache))
        docmodels = {}
        doclength_dict = {}

        for docname in tqdm(sorted(os.listdir(transcript_dir))):
            # Language model dictionary, store for inverted index and doclength
            docmodels[ docname ] = defaultdict(float)
            doc_path = os.path.join(transcript_dir,docname)
            # Read transcription
            with open(doc_path,'r') as f:
                text = f.read()
                uni_text = text.decode('utf-8')
                split_uni_text = ''.join(uni_text.split()) # Remove whitespace & endline
                cut_uni_text = list(jieba.cut(split_uni_text)) # Split with jieba

                doclength_dict[ docname ] = len(cut_uni_text)

                # Unicode to bracket big5 hex
                for word in cut_uni_text:
                    # To big5 hex
                    big5_word = word.encode('big5')
                    big5_hex = ""
                    for c in big5_word:
                        big5_hex += format(ord(c),'02X')

                    # Add brackets
                    bracketed_chars = ''
                    assert len(big5_hex) % 4 == 0
                    for i in range(0,len(big5_hex),4):
                        bracketed_chars += '[' + big5_hex[i:i+4] + ']'

                    if bracketed_chars in lex_dict:
                        lex_index = lex_dict[ bracketed_chars ]
                        docmodels[ docname ][ lex_index ] += 1.

                # Normalize docmodel
                factor = 1. / sum( docmodels[ docname ].values() )
                for k in docmodels[ docname ].keys():
                    docmodels[ docname ][ k ] *= factor

        with open(docmodels_cache,'w') as f:
            obj = (docmodels, doclength_dict)
            pickle.dump(obj, f)

    else:
        print("Loading docmodel cache from {}".format(docmodels_cache))
        with open(docmodels_cache,'r') as f:
            docmodels, doclength_dict = pickle.load(f)
    # Read from transcript dir, save to docmodel dir
    if not os.path.exists(save_docmodel_dir):
        print("Creating dir at {}".format(save_docmodel_dir))
        os.makedirs(save_docmodel_dir)

    # Write language model for every document
    print("Writing language models to {}".format(save_docmodel_dir))
    for docname in tqdm(os.listdir(transcript_dir)):
        if docname in docmodels.keys():
            save_docname = os.path.join(save_docmodel_dir,docname)
            with open(save_docname,'w') as fout:
                for k, v in docmodels[ docname ].iteritems():
                    fout.write('{} {}\n'.format(k,v))

    # Write doclength file
    print("Writing doclength file to {}".format(doclength_file))
    with open(doclength_file,'w') as fout:
        for k in tqdm(sorted(doclength_dict.keys())):
            fout.write("{} {}\n".format(k, doclength_dict[k]))

    # Write inverted index
    print("Writing inverted index to {}".format(index_file))
    with open(index_file,'w') as fout:
        for word_index in tqdm(range(1,len(lex_dict),1)):
            inv_index_string = ""
            # Loop through all the documents
            sorted_docmodel_keys = sorted(docmodels.keys())
            for doc_index, key in enumerate(sorted_docmodel_keys,1):
                docmodel = docmodels[ key ]
                if word_index in docmodel:
                    word_prob = docmodel[ word_index ]
                    inv_index_string += '{}:{}'.format(doc_index,word_prob)

            fout.write("{}\t{}\n".format(word_index,inv_index_string))



if __name__ == "__main__":
    # Already existed
    PTV_dir = './data/ISDR-CMDP'
    lex_file = os.path.join(PTV_dir,'PTV.lex')

    # Self defined paths
    transcript_name = 'dnn'
    transcript_dir = './data/PTV_onebest_fromMATBN_charSeg'

    docmodels_cache = os.path.join(transcript_dir,'docmodels.pickle')

    save_docmodel_dir = os.path.join(PTV_dir,'docmodel',transcript_name)

    doclength_file = os.path.join(PTV_dir,'doclength',transcript_name+'.length')
    index_file = os.path.join(PTV_dir,'index',transcript_name+'.index')

    # To write:
    # 1. transcript
    # 2. doclength
    # 3. inverted index

    # Results paths

    run_transcript2docmodel(lex_file, transcript_dir, docmodels_cache, save_docmodel_dir, doclength_file, index_file)
