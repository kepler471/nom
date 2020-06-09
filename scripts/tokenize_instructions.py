import json
import numpy as np
import pickle
import sys
from tqdm import *
import time
import os
import ijson
from itertools import islice


def readfile(filename):
    with open(filename,'r') as f:
        lines = []
        for line in f.readlines():
            lines.append(line.rstrip())
    return lines

def tok(text,ts=False):

    '''
    Usage: tokenized_text = tok(text,token_list)
    If token list is not provided default one will be used instead.
    '''

    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']

    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text

if __name__ == "__main__":

    '''
    Generate tokenized text for w2v training
    Words separated with ' '
    Different instructions separated with \t
    Different recipes separated with \n
    '''

    try:
        partition = str(sys.argv[1])
    except:
        partition = ''

    try:
        num_to_load = int(sys.arg[2])
    except KeyError:
        num_to_load = 200

    dets_loc = os.path.dirname(os.path.abspath(__file__)) + '/../data/recipe1M/det_ingrs.json'
    layer1_loc = os.path.dirname(os.path.abspath(__file__)) + '/../data/recipe1M/layer1.json'
    dets_iterator = ijson.parse(open(dets_loc,'r'))
    layer1_iterator = ijson.parse(open(layer1_loc,'r'))

    with open(layer1_loc) as f:
        item = ijson.items(f, 'item')
        objects = islice(item, num_to_load)
        layer1 = [item for item in objects]

    with open(dets_loc) as f:
        item = ijson.items(f, 'item')
        objects = islice(item, num_to_load)
        dets = [item for item in objects]

    idx2ind = {}
    ingrs = []
    for i,entry in enumerate(dets):
        idx2ind[entry['id']] = i


    t = time.time()
    print("Saving tokenized here:", '../data/tokenized_instructions_'+partition+'.txt')
    f = open('../data/tokenized_instructions_'+partition+'.txt','w')
    for i,entry in tqdm(enumerate(layer1)):
        '''
        if entry['id'] in dups:
            continue
        '''
        if not partition=='' and not partition == entry['partition']:
            continue
        instrs = entry['instructions']

        allinstrs = ''
        for instr in instrs:
            instr =  instr['text']
            allinstrs+=instr + '\t'

        # find corresponding set of detected ingredients
        det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
        valid = dets[idx2ind[entry['id']]]['valid']

        for j,det_ingr in enumerate(det_ingrs):
            # if detected ingredient matches ingredient text,
            # means it did not work. We skip
            if not valid[j]:
                continue
            # underscore ingredient

            det_ingr_undrs = det_ingr['text'].replace(' ','_')
            ingrs.append(det_ingr_undrs)
            allinstrs = allinstrs.replace(det_ingr['text'],det_ingr_undrs)

        f.write(allinstrs + '\n')

    f.close()
    print(time.time() - t, 'seconds.')
    print("Number of unique ingredients",len(np.unique(ingrs)))
    f = open('../data/tokenized_instructions_'+partition+'.txt','r')
    text = f.read()
    text = tok(text)
    f.close()

    f = open('../data/tokenized_instructions_'+partition+'.txt','w')
    f.write(text)
    f.close()
