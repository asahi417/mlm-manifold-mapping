""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

import transformers
from bertprompt import get_lama_data


class Test(unittest.TestCase):
    """ Test """

    def test(self):

        t = transformers.AutoTokenizer.from_pretrained('roberta-large')
        get_lama_data(vocab=t.vocab)

        t = transformers.AutoTokenizer.from_pretrained('bert-large-cased')
        get_lama_data(vocab=t.vocab)

    def test_transformer(self):
        get_lama_data(transformers_model=['roberta-large', 'bert-large-cased'])


if __name__ == "__main__":
    unittest.main()
