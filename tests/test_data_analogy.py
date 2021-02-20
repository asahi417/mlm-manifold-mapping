""" UnitTest """
import unittest
import logging
from bertprompt import get_analogy_data

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test_transformer(self):
        get_analogy_data('u2')


if __name__ == "__main__":
    unittest.main()
