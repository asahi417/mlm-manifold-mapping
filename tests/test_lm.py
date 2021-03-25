""" UnitTest """
import unittest
import logging
from pprint import pprint
from bertprompt import Prompter
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    # def test(self):
    #     lm = Prompter('albert-base-v1', max_length=32)
    #     # lm = Prompter('bert-large-cased', max_length=32)
    #
    #     # test word pair infilling
    #     test_candidates = [["mask", "hedonist"], ["emotion", "demagogue"], ["dog", "cat"], ['vapor', 'evaporate']]
    #     pprint(lm.generate(word_pairs=test_candidates, n_blank=2, n_blank_b=0, n_blank_e=0, n_revision=1))
    #
    #     # test sentence revision
    #     test_sentences = ['emotion and violence: demagogue', 'opinion of a communist sympathizer']
    #     vocab_to_keep = ['demagogue', 'opinion']  #
    #     pprint(lm.generate(seed_sentences=test_sentences, vocab_to_keep=vocab_to_keep, n_revision=2))
    #
    #     # test sentence revision with mask
    #     test_sentences = ['One of the things you do when you are alive is [MASK].',
    #                       'Something that might happen while analysing something is [MASK].',
    #                       'Competing against someone requires a desire to [MASK]',
    #                       'The fact "one whale plus one whale plus one whale equals three whales "'
    #                       ' is illustrated with the story:1. Whales are marine animals.2. Adding'
    #                       ' "plus" means adding to the [MASK] of numbers.3. One + one + one equals three.4. '
    #                       'Three is a number.5. Numbers are the total addition of units taken together.']
    #     vocab_to_keep = [['alive', '[MASK]'], ['[MASK]'], ['competing', '[MASK]'], ['number', '[MASK]']]
    #     pprint(lm.generate(seed_sentences=test_sentences, vocab_to_keep=vocab_to_keep, n_revision=2, topk=15))
    #
    # def test_issue1(self):
    #     lm = Prompter('albert-base-v1', max_length=64)
    #     # lm = Prompter('bert-large-cased', max_length=32)
    #     test_sentences = [' "plus" means adding to the [MASK] of numbers.3. One + one + one equals three.4. Three is a number.5. Numbers']
    #     vocab_to_keep = [['number', '[MASK]']]
    #     pprint(lm.generate(seed_sentences=test_sentences, vocab_to_keep=vocab_to_keep,
    #                        vocab_to_keep_unique=False,
    #                        n_revision=1, topk=1))
    #
    # def test_issue2(self):
    #     lm = Prompter('albert-base-v1', max_length=64)
    #     pprint(lm.generate(word_pairs=['advertisement', 'agenda'],
    #                        n_revision=1,
    #                        n_blank=1,
    #                        n_blank_b=0,
    #                        n_blank_e=0,
    #                        topk=1))
    #     pprint(lm.generate(seed_sentences=['advertisement agenda'],
    #                        vocab_to_keep=[['advertisement', 'agenda']],
    #                        n_revision=1,
    #                        topk=1))
    #
    # def test_issue3(self):
    #     lm = Prompter('albert-base-v1', max_length=32)
    #     test_sentences = ['Interleukin 6 signal transducer (Gp130, oncostatin M receptor) is a subclass of [MASK]']
    #     vocab_to_keep = [['[MASK]', 'Interleukin 6 signal transducer (Gp130, oncostatin M receptor)']]
    #     pprint(lm.generate(seed_sentences=test_sentences, vocab_to_keep=vocab_to_keep,
    #                        vocab_to_keep_unique=False,
    #                        n_revision=1,
    #                        topk=1))
    #
    # def test_issue4(self):
    #     lm = Prompter('albert-base-v1', max_length=32)
    #     test_sentences = ['Stanis%C5%82aw Stolarczyk was born in [MASK] .']
    #     vocab_to_keep = [['Stanis%C5%82aw Stolarczyk', '[MASK]']]
    #     pprint(lm.generate(seed_sentences=test_sentences,
    #                        vocab_to_keep=vocab_to_keep,
    #                        vocab_to_keep_unique=False,
    #                        n_revision=1,
    #                        topk=1))

    # def test_issue5(self):
    #     lm = Prompter('albert-base-v1', max_length=64)
    #     pprint(lm.generate(seed_sentences=['advertisement [MASK]'],
    #                        vocab_to_keep=[['advertisement', '[MASK]']],
    #                        n_revision=1,
    #                        topk=1))
    #     pprint(lm.generate(seed_sentences=['advertisement ask'],
    #                        vocab_to_keep=[['advertisement', 'ask']],
    #                        n_revision=1,
    #                        topk=1))

    # def test_issue5(self):
    #     lm = Prompter('albert-base-v1', max_length=64)
    #     pprint(lm.generate(seed_sentences=['Visual Basic .NET is developed by [MASK] .'],
    #                        vocab_to_keep=[['[MASK]', 'Visual Basic .NET']],
    #                        n_revision=1,
    #                        topk=1))
    #     lm = Prompter('bert-large-cased', max_length=64)
    #     pprint(lm.generate(seed_sentences=['Csaba Őry was born in [MASK] .'],
    #                        vocab_to_keep=[['[MASK]', 'Csaba Őry']],
    #                        n_revision=1,
    #                        topk=1))

    def test_issue6(self):
        # lm = Prompter('albert-base-v1', max_length=64)
        lm = Prompter('roberta-large', max_length=64)
        pprint(lm.generate(word_pairs=['Albania', 'Albanian'],
                           n_revision=1,
                           n_blank=2,
                           n_blank_b=0,
                           n_blank_e=0,
                           topk=15))
        pprint(lm.generate(word_pairs=['aberdeen', 'aberdeenshire'],
                           n_revision=1,
                           n_blank=2,
                           n_blank_b=0,
                           n_blank_e=0,
                           vocab_to_keep_unique=True,
                           topk=15))



if __name__ == "__main__":
    unittest.main()
