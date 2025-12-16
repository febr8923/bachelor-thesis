from bert import QA
from timeit import default_timer as timer

before = timer()
model = QA('model')
after = timer()

print('model eval time:')
print(after - before)

for i in range(5):

    start = timer()

    doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
        "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
        "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
        "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
        "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
        "people in a referendum, depending on the provision. "

    q = 'When did Victoria enact its constitution?'

    answer = model.predict(doc, q)
    print(answer['answer'])
# print(answer.keys())

    end = timer()
    print(end - start)

import signal, time

def signal_handler(signum, frame):
    print("\nSignal received. Exiting gracefully.")
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

print("Running... Press Ctrl+C to exit.")

try:
    # This will run indefinitely until a SIGINT is received
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # This block will not be executed due to our signal handler
    pass

print("This line will not be reached.")
