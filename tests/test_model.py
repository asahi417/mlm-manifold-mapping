import logging
from pprint import pprint
from m3 import Rewriter

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

sample_sentence = [
    "AHH i'm so HAPPY. I just found my ipod. God is sooo good to me!",
    "I remember getting this book so faintly that that says alot about my opinion of it.",
    "Basically, while I will entertain lots of odd ideas and theories, this book was basically silly."
]


if __name__ == '__main__':
    model = Rewriter(max_length=64)
    out = model.generate(sample_sentence, max_n_iteration=3, topk=3)
    pprint(out)
