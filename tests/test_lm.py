""" UnitTest """
import unittest
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from bertprompt import Prompter


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        lm = Prompter('albert-base-v1', max_length=16)

        # test word pair infilling
        test_candidates = [["pleasure", "hedonist"], ["emotion", "demagogue"], ["opinion", "sympathizer"]]
        pprint(lm.generate(word_pairs=test_candidates, n_blank=2, n_blank_b=1, n_blank_e=1, debug=True))

        # test sentence revision
        test_sentences = ['emotion and violence: demagogue', 'opinion of a communist sympathizer']
        pprint(lm.generate(seed_sentences=test_sentences, debug=True))


if __name__ == "__main__":
    unittest.main()
