import re


def check_vocab(sentence, vocab):
    vocab_in = re.findall(r'|'.join(vocab), sentence)
    vocab_in_unique = list(set(vocab_in))
    if len(vocab_in_unique) == len(vocab_in) == len(vocab):
        return True
    return False


tests = [
    ['cat cats dogs dog', ['cat', 'dog'], False],
    ['dogs dogs', ['cat', 'dog'], False],
    ['evaporate is vapor', ['evaporate', 'vapor'], True],
    ['jeer s mock-up', ['jeer', 'mock'], True]
]

for s, v, expect in tests:
    flag = check_vocab(s, v)
    assert expect == flag, (s, v, flag)
