""" Masked Language Model based Prompting """
import re
import os
import logging
import math
from itertools import chain
from typing import List
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool

import torch
from torch import nn

from .util import load_model, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


def pool_map(f, arg):
    """ Multiprocessing map function. """
    _pool = Pool()
    out = _pool.map(f, arg)
    _pool.close()
    return out


def get_partition(_list):
    """ Get partition in multiprocess. """
    p = Partition(_list)
    return pool_map(p, range(len(_list)))


def get_encoding(sentences, tokenizer, max_length):
    """ Get encode_plus in multiprocess. """
    p = EncodePlus(tokenizer, max_length)
    return pool_map(p, sentences)


class Partition:
    """ Get the partition information of a nested list for restoring the original structure. """

    def __init__(self, _list):
        self.length = pool_map(len, _list)

    def __call__(self, x):
        return [sum(self.length[:x]), sum(self.length[:x + 1])]


class EncodePlus:
    """ Get encode_plus output in parallel. """

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        if max_length:
            assert self.max_length >= max_length, f'{self.max_length} < {max_length}'
            self.max_length = max_length
        # sentence prefix tokens to fix offset when encoding sentence
        tokens = self.tokenizer.tokenize('get tokenizer specific prefix')
        tokens_encode = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('get tokenizer specific prefix'))
        self.sp_token_prefix = tokens_encode[:tokens_encode.index(tokens[0])]

    def input_ids_to_labels(self,
                            input_ids: List,
                            label_position: List = None,
                            label_id: List = None):
        """ Generate a label for language model loss. If `label_position` is None, label is the original input_ids for
        every token except padding token, or it keeps what specified by `label_position` with label defined by
        `label_id`.

        Parameters
        ----------
        input_ids : list
            The input_ids given by tokenizer.encode .
        label_position : list
            Position in input_ids for prediction.
        label_id :
            Token id for each position in `label_position`.

        Returns
        -------
        List of `label` that can be used for loss computation
        """
        if label_position is None and label_id is None:  # just to ignore padding token
            label = list(map(lambda x: PAD_TOKEN_LABEL_ID if x == self.tokenizer.pad_token_id else x, input_ids))
        else:
            assert len(label_position) == len(label_id), f'{len(label_position)} != {len(label_id)}'
            label = [PAD_TOKEN_LABEL_ID] * len(input_ids)
            for p, i in zip(label_position, label_id):
                label[p] = i
        return label

    def __call__(self, sentence):
        """ Encoding sentence with label that is
        - masked token if `token_wise_mask` is False (mainly for token prediction)
        - otherwise every token that is not mask token (mainly for perplexity computation)

        Parameters
        ----------
        sentence : str
            A string sentence.

        Returns
        -------
        A list of the output from tokenizer.encode_plus .
        """
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        token_list = self.tokenizer.tokenize(sentence)

        def encode_with_single_mask_id(mask_position: int):
            _token_list = token_list.copy()  # can not be encode outputs because of prefix
            masked_token_id = self.tokenizer.convert_tokens_to_ids(_token_list[mask_position])
            if masked_token_id == self.tokenizer.mask_token_id:
                return None
            _token_list[mask_position] = self.tokenizer.mask_token
            tmp_string = self.tokenizer.convert_tokens_to_string(_token_list)
            _encode = self.tokenizer.encode_plus(tmp_string, **param)
            assert _encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded max_length'
            _encode['labels'] = self.input_ids_to_labels(
                _encode['input_ids'],
                label_position=[mask_position + len(self.sp_token_prefix)],
                label_id=[masked_token_id])
            return _encode

        length = min(self.max_length - len(self.sp_token_prefix), len(token_list))
        return list(filter(None, map(encode_with_single_mask_id, range(length))))


class Rewriter:
    """ Rewrite sentence based on manifold mapping onto the low-perplexity space. """

    def __init__(self,
                 model: str = 'albert-base-v2',
                 max_length: int = 32,
                 num_worker: int = 0):
        """ Rewrite sentence based on manifold mapping onto the low-perplexity space.

        Parameters
        ----------
        model : str
            Transformers model (eg. distilbert-base-uncased, distilbert-base-cased, albert-base-v2).
        max_length : int
            A model max length if specified, else use model_max_length.
        num_worker : int
        """
        logging.info('Initialize Rewriter')
        # assert 'bert' in model, f'{model} is not BERT'
        self.num_worker = num_worker
        self.model_name = model
        self.max_length = max_length
        try:
            self.tokenizer, self.config, self.model, self.device, self.parallel = load_model(model, local_files_only=False)
        except ValueError:
            self.tokenizer, self.config, self.model, self.device, self.parallel = load_model(model, local_files_only=True)

    def cleanup_decode(self, sentence):
        """ Clean up sentence with tokenizers special tokens """
        mask = self.tokenizer.mask_token
        # give a space around mask token
        cleaned_sent = re.sub(r'({})'.format(mask).replace('[', '\[').replace(']', '\]'), r' \1 ', sentence)
        # reduce more than two spaces to one
        cleaned_sent = re.sub(r'\s+', ' ', cleaned_sent)
        # remove special tokens but keep mask
        to_remove = list(filter(lambda x: x != mask, self.tokenizer.all_special_tokens))
        cleaned_sent = re.sub(r'|'.join(list(map(re.escape, to_remove))), '', cleaned_sent)
        # remove redundant spaces on the beginning of the sentence
        return re.sub(r'\A\s*', '', cleaned_sent)

    def generate(self,
                 seed_sentences: List = None,
                 vocab_to_keep: List = None,
                 vocab_to_keep_unique: bool = False,
                 max_n_iteration: int = 100,
                 topk: int = 10,
                 topk_buffer: int = 100,
                 batch_size: int = 4):
        """ Generate/rewrite prompt based on perplexity

        Parameters
        ----------
        seed_sentences : list
        vocab_to_keep : list
            (optional) A list of token to keep while replacing.
        vocab_to_keep_unique : bool
            (optional) Allow to include words from `vocab_to_keep` only once.
        max_n_iteration : the number of revision after replacing all the mask
        batch_size : int
        topk : int
            Keep topk prediction on masked token for perplexity filtering.
        topk_buffer : int
            Number of token to compute before top-k.
        """
        tol = 0.05
        if type(seed_sentences) is str:
            seed_sentences = [seed_sentences]
        assert all(map(len, seed_sentences)), f'empty string found in {seed_sentences}'
        ppl = self.get_perplexity(seed_sentences, batch_size=batch_size)
        edit = [[s] for s in seed_sentences]
        edit_ppl = [[s] for s in ppl]
        data_key = {k: v for k, v in enumerate(seed_sentences)}
        logging.info('### ITERATIVE REVISION ###')
        data_index = list(data_key.keys())
        output_list = [[]] * len(data_index)
        i = 0
        while True:
            if i > max_n_iteration:
                logging.info('ITERATIVE REVISION: reached max revision step')
                break
            logging.info(f'ITERATIVE REVISION: step {i + 1} (max {max_n_iteration} steps)')
            seed_sentences, ppl = self.replace_single_token(
                seed_sentences,
                vocab_to_keep=vocab_to_keep,
                vocab_to_keep_unique=vocab_to_keep_unique,
                topk=topk,
                topk_buffer=topk_buffer,
                batch_size=batch_size,
            )

            # sentence keep improving
            index_unfixed = list(filter(lambda x: (edit_ppl[x][-1] - ppl[x]) > tol, range(len(seed_sentences))))

            # extract stable sentence
            index_fixed = list(filter(lambda x: x not in index_unfixed, range(len(seed_sentences))))
            for n in index_fixed:
                output_list[data_index[n]] = [edit[n], edit_ppl[n]]

            edit = list(map(lambda x: tuple(list(edit[x]) + [seed_sentences[x]]), index_unfixed))
            edit_ppl = list(map(lambda x: tuple(list(edit_ppl[x]) + [ppl[x]]), index_unfixed))
            seed_sentences = list(map(lambda x: seed_sentences[x], index_unfixed))
            ppl = list(map(lambda x: ppl[x], index_unfixed))
            data_index = list(map(lambda x: data_index[x], index_unfixed))
            if vocab_to_keep:
                vocab_to_keep = list(map(lambda x: vocab_to_keep[x], index_unfixed))

            if len(seed_sentences) == 0:
                logging.info('ITERATIVE REVISION: all sentences reached the best perplexity')
                break
            i += 1

        for i in range(len(data_index)):
            output_list[data_index[i]] = [edit[i], edit_ppl[i]]
        return output_list

    def replace_single_token(self,
                             seed_sentences: List,
                             vocab_to_keep: List = None,
                             vocab_to_keep_unique: bool = False,
                             batch_size: int = 4,
                             topk: int = 5,
                             topk_buffer: int = 100):
        """ Replace single token (run parallel over given lists)
        - (i) Greedy token prediction: predict token by masking each token or masked token if sentence consists of mask
        - (ii) Perplexity re-ranking: choose the best replaced sentence that achieves the best perplexity

        Parameters
        ----------
        seed_sentences : list
        vocab_to_keep : list
            (optional) A list of token to keep while replacing.
        vocab_to_keep_unique : bool
            (optional) Allow to include words from `vocab_to_keep` only once.
        batch_size : int
        topk : int
            Keep topk prediction on masked token for perplexity filtering.
        topk_buffer : int
            Number of token to compute before top-k.
        """

        def check_vocab(sentence, vocab):
            if not vocab_to_keep_unique:
                return True
            vocab_unique = list(set(vocab))
            vocab_unique = sorted(vocab_unique, reverse=True)
            vocab_in = re.findall(r'|'.join(vocab_unique).lower(), sentence.lower())
            vocab_in_unique = list(set(vocab_in))
            if len(vocab_in_unique) == len(vocab_in) == len(vocab):
                return True
            elif len(vocab_unique) != len(vocab):
                if len(vocab) == len(vocab_in) and len(vocab_unique) == len(vocab_unique):
                    return True
            return False

        if vocab_to_keep:
            assert len(seed_sentences) == len(vocab_to_keep), f'{len(seed_sentences)} != {len(vocab_to_keep)}'

        if type(seed_sentences) is str:
            seed_sentences = [seed_sentences]

        data = get_encoding(seed_sentences, self.tokenizer, self.max_length)
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)

        logging.info('\t* prediction on masked tokens')
        total_input, total_val, total_ind, labels = [], [], [], []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                values, indices = prediction_scores.topk(topk_buffer, dim=-1)
                total_input += encode.pop('input_ids').tolist()
                total_val += values.tolist()
                total_ind += indices.tolist()
                if 'labels' in encode:
                    labels += encode.pop('labels').tolist()

        greedy_filling = []
        logging.info(f'\t* filter to top {topk} prediction')
        for partition_n, (s, e) in enumerate(tqdm(partition)):
            v = None
            v_mask = False
            if vocab_to_keep:
                # convert all tokens from keep_vocab to suitable form of the tokenizer
                v = vocab_to_keep[partition_n]
                if type(v) is str:
                    v = [v]
                if self.tokenizer.mask_token in v:
                    # mask token is handled separately
                    v = v.copy()
                    v.pop(v.index(self.tokenizer.mask_token))
                    v_mask = True
                # convert keep_vocab in a way tokenizer can deal with
                v = [self.tokenizer.decode(self.tokenizer(v_)['input_ids'], skip_special_tokens=True).lower()
                     for v_ in v]
                # add escape symbol to special character so that it not fails in regx
                v = list(map(re.escape, v))
                # if sentence only has tokens from vocab_to_keep, skip process
                sent = seed_sentences[partition_n]
                if v_mask:
                    sent = re.sub(
                        r'|'.join(v + [re.escape(self.tokenizer.mask_token.lower())]), '',
                        sent.lower()).replace(' ', '')
                else:
                    sent = re.sub(r'|'.join(v), '', sent.lower()).replace(' ', '')
                if len(sent) == 0:
                    greedy_filling.append([seed_sentences[partition_n]])
                    continue

            def process_single_pair(_topk, allow_subword=False):
                topk_decoded = []
                for i in range(s, e):
                    inp, val, ind = total_input[i], total_val[i], total_ind[i]
                    if labels:
                        label = labels[i]
                        filtered = list(filter(
                            lambda x: label[x[0]] != PAD_TOKEN_LABEL_ID, enumerate(zip(val, ind))))
                    else:
                        filtered = list(filter(
                            lambda x: inp[x[0]] == self.tokenizer.mask_token_id, enumerate(zip(val, ind))))

                    def decode_topk(k, replace_pos, token_index, token_likelihood):
                        tokens = deepcopy(inp)
                        tokens[replace_pos] = token_index[k]
                        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
                        decoded = self.cleanup_decode(decoded)
                        decoded_no_mask = decoded.replace(self.tokenizer.mask_token, '')
                        if v:
                            if allow_subword:
                                # very important to apply re.escape, otherwise it gets error if x contains special
                                # characters such as ()[]\.
                                if not all(map(lambda x: len(re.findall(r'{}'.format(x), decoded_no_mask.lower())), v)):
                                    return None
                            else:
                                if not all(map(lambda x: len(re.findall(
                                        r'\b{}\b'.format(x), decoded_no_mask.lower())), v)):
                                    return None

                            # check if all tokens from keep_vocab just appeared once

                            if not check_vocab(decoded_no_mask, v):
                                return None
                        if v_mask and self.tokenizer.mask_token not in decoded:
                            return None

                        return decoded, token_likelihood[k]

                    for _replace_pos, (_val, _ind) in filtered:
                        topk_decoded += list(filter(
                            None, map(lambda x: decode_topk(x, _replace_pos, _ind, _val), range(_topk))
                        ))
                return topk_decoded

            topk_edit = process_single_pair(topk)
            if len(topk_edit) == 0 and topk_buffer > topk:
                topk_edit = process_single_pair(topk_buffer)
            if len(topk_edit) == 0:
                topk_edit = process_single_pair(topk_buffer, True)
                if len(topk_edit) != 0 and v is not None:
                    logging.warning('prompt may include subword (ignore if term to keep consists of multiple words)')
                    logging.warning(f'\t - prompt      : {topk_edit[0]}')
                    logging.warning(f'\t - term to keep: {v}')

            if len(topk_edit) == 0:
                raise ValueError(f'no valid sentence found: ({vocab_to_keep[partition_n]})'
                                 f'\n- current prompt: {seed_sentences[partition_n]}')
            # drop duplicated decode and keep the one with the highest likelihood
            topk_edit = list(map(
                lambda d: max(filter(lambda x: x[0] == d, topk_edit), key=lambda x: x[1]),
                set(list(zip(*topk_edit))[0])
            ))
            topk_edit = sorted(topk_edit, key=lambda x: x[1], reverse=True)
            topk_candidate = list(zip(*topk_edit))[0][:min(topk, len(topk_edit))]
            greedy_filling.append(topk_candidate)
        logging.info('\t* ppl re-ranking')
        partition = get_partition(greedy_filling)
        list_ppl = self.get_perplexity(list(chain(*greedy_filling)), batch_size=batch_size)
        list_ppl = [list_ppl[s:e] for s, e in partition]
        best_edit = []
        best_ppl = []
        for sent, ppl in zip(greedy_filling, list_ppl):
            best_edit.append(sent[ppl.index(min(ppl))])
            best_ppl.append(min(ppl))

        logging.info('\t* edit sample')
        for n, (o, ed, bp) in enumerate(zip(seed_sentences, best_edit, best_ppl)):
            logging.info(f'\t\t- original: {o}')
            logging.info(f'\t\t- edit    : {ed} (ppl: {bp})')
            if n > 5:
                break
        return best_edit, best_ppl

    def get_perplexity(self, sentences, batch_size: int = 4):
        """ Compute perplexity on sentences

        Parameters
        ----------
        sentences : list
        batch_size : int

        Returns
        ----------
        A list of perplexity.
        """
        if type(sentences) is str:
            sentences = [sentences]
        data = get_encoding(sentences, self.tokenizer, self.max_length)
        partition = get_partition(data)
        data_loader = torch.utils.data.DataLoader(
            Dataset(list(chain(*data))),
            num_workers=self.num_worker,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll = []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                labels = encode.pop('labels')
                output = self.model(**encode, return_dict=True)
                prediction_scores = output['logits']
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss.view(len(prediction_scores), -1)
                loss = torch.sum(loss, -1)
                nll += list(map(
                    lambda x: x[0] / sum(map(lambda y: y != PAD_TOKEN_LABEL_ID, x[1])),
                    zip(loss.cpu().tolist(), labels.cpu().tolist())
                ))
        return list(map(lambda x: math.exp(sum(nll[x[0]:x[1]]) / (x[1] - x[0])), partition))
