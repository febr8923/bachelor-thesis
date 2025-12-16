
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)

from bert import QA
from timeit import default_timer as timer

import torch

model = None

INPUT_DIR = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, os.path.pardir, 'benchmark-inputs', 'ml-inference', 'BERT-SQuAD')

def function(obj):

    global model

    if model is None:

        #print("model", flush=True)
        #before = timer()
        model = QA(os.path.join(INPUT_DIR, 'model'))
        #after = timer()
        #print("model end", flush=True)

        #print('model eval time:', after - before, flush=True)

    start = timer()

    doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
        "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
        "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
        "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
        "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
        "people in a referendum, depending on the provision. "

    q = 'When did Victoria enact its constitution?'

    #print('model predict', flush=True)
    answer = model.predict(doc, q)
    print(answer['answer'], flush=True)
# print(answer.keys())

    end = timer()
    #print(end - start)

if __name__ == "__main__":
    for i in range(11):
        start = timer()
        function({})
        end = timer()
        print("Time:", end-start)
