import os
import re
import sys
import random
import pickle
import argparse
from collections import defaultdict
import torch


random.seed(1234)


class DataHandler:
    """
    Data format:
    ex: { 'pi:num':
            {'protoform':
                {
                'Latin': ['p', 'i', 'n', 'ʊ', 'm']
                },
            'daughters':
                {'Romanian': ['p', 'i', 'n'],
                 'French': ['p', 'ɛ', '̃'],
                 'Italian': ['p', 'i', 'n', 'o'],
                 'Spanish': ['p', 'i', 'n', 'o'],
                 'Portuguese': ['p', 'i', 'ɲ', 'ʊ']
                }
            },
        ...
        }
    """
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name

    def _read_tsv(self, fpath):
        """
        Assumes the first row contains the languages (daughter and proto-lang)
        Assumes the first column is the protoform (or characters in the case of Chinese)

        Returns a dict mapping a protoform to the daughter forms
        """
        with open(fpath) as fin:
            langs = fin.readline().strip().split('\t')
            if "chinese" in self._dataset_name:
                langs = langs[1:]  # first column is character
            d = {}
            for line in fin:
                tkns = line.strip().split('\t')
                d[tkns[0]] = tkns[1:]
        return langs, d

    def _clean_middle_chinese_string(self, clean_string):
        subtokens = clean_string.split('/')
        tone = None
        if len(subtokens) > 1:
            tone = subtokens[1]
            tone = {
                '¹': '平',
                '²': '上',
                '³': '去',
                '⁴': '入'
            }[tone]
            return subtokens[0], tone
        else:
            # the tone is not separated by a / - tɕʰoŋʷ¹
            return clean_string[:-1], {
                '¹': '平',
                '²': '上',
                '³': '去',
                '⁴': '入'
            }[clean_string[-1]]


    def _clean_sinitic_daughter_string(self, raw_string):
        # only keep first entry for multiple variants (polysemy, pronunciation variation, etc.)
        # selection is arbitrary -> can also be removed altogether
        clean_string = raw_string
        if '|' in raw_string:
            clean_string = raw_string.split('|')[0]
        if '/' in raw_string:
            clean_string = raw_string.split('/')[0]
        # remove chinese characters
        subtokens = re.findall('([^˩˨˧˦˥]+)([˩˨˧˦˥]+)', clean_string)
        tone = None
        if subtokens:
            subtokens = subtokens[0]
            clean_string = subtokens[0]
            tone = subtokens[1]
        return clean_string, tone

    def sinitic_tokenize(self, clean_string, merge_diacritics=False):
        tkns = list(clean_string)

        ### TODO: middle chinese didn't connect affricates with '͡' --> will remove these tokens for now.
        # affricate - should always be merged
        while '͡' in tkns:
            i = tkns.index('͡')
            tkns = tkns[:i-1] + [''.join(tkns[i-1: i+2])] + tkns[i+2:]

        tkns = [tkn for tkn in tkns if tkn != '͡']

        # diacritics - optionally merge
        if merge_diacritics:
            diacritics = {'ː', '̃', '̍', '̞', '̠', '̩' , 'ʰ', 'ʷ', '̱'}
            # TODO: should diphthong be merged? yes, eventually
            while diacritics & set(tkns):
                for i in range(len(tkns)):
                    # if tkns[i] == '̯': # infinite loop
                    #     tkns = tkns[:i-2] + [''.join(tkns[i-2: i+1])] + tkns[i+1:]
                    #     break
                    if tkns[i] in diacritics:
                        tkns = tkns[:i-1] + [''.join(tkns[i-1: i+1])] + tkns[i+1:]
                        break
        return tkns

    def tokenize(self, string):
        return list(string)

    def generate_split_datasets(self):
        split_ratio = (70, 10, 20)  # train, dev, test
        langs, data = self._read_tsv(f'./data/{self._dataset_name}.tsv')
        protolang = langs[0]
        cognate_set = {}
        for cognate, tkn_list in data.items():
            entry = {}
            daughter_sequences = {}
            if "chinese" in self._dataset_name:
                mc_string, mc_tone = self._clean_middle_chinese_string(tkn_list[0])
                # we assume there is always a tone for the MC string
                mc_tkns = self.sinitic_tokenize(mc_string, merge_diacritics=True) + [mc_tone]
                for dialect, tkn in zip(langs[1:], tkn_list[1:]):
                    if not tkn or tkn == '-':
                        continue
                    daughter_string, daughter_tone = self._clean_sinitic_daughter_string(tkn)
                    daughter_tkns = self.sinitic_tokenize(daughter_string, merge_diacritics=True)
                    if daughter_tone:
                        daughter_tkns += [daughter_tone]
                    daughter_sequences[dialect] = daughter_tkns
                entry['protoform'] = {
                    protolang: mc_tkns
                }
                entry['daughters'] = daughter_sequences
                cognate_set[cognate] = entry
            else:
                protolang_tkns = self.tokenize(cognate)
                for lang, tkn in zip(langs[1:], tkn_list):
                    if not tkn or tkn == '-':
                        continue
                    daughter_tkns = self.tokenize(tkn)
                    daughter_sequences[lang] = daughter_tkns

                entry['protoform'] = {
                    protolang: protolang_tkns
                }
                entry['daughters'] = daughter_sequences
                cognate_set[cognate] = entry

        dataset = {}
        proto_words = list(cognate_set.keys())
        random.shuffle(proto_words)
        dataset['train'] = proto_words[0: int(len(proto_words) * split_ratio[0]/sum(split_ratio))]
        dataset['dev'] = proto_words[len(dataset['train']): int(len(proto_words) * (split_ratio[0] + split_ratio[1])/sum(split_ratio))]
        dataset['test'] = proto_words[len(dataset['train']) + len(dataset['dev']): ]

        dataset_path = f'./data/{self._dataset_name}'
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        for data_type in dataset:
            subdata = {protoword: cognate_set[protoword] for protoword in dataset[data_type]}
            with open(f'./data/{self._dataset_name}/{data_type}.pickle', 'wb') as fout:
                pickle.dump((langs, subdata), fout)

    @classmethod
    def load_dataset(cls, fpath):
        vocab = set()  # set of possible phonemes in the daughters and the protoform
        with open(fpath, 'rb') as fin:
            (langs, data) = pickle.load(fin)
            # list format enables shuffling
            dataset = []
            for cognate, entry in data.items():
                for lang, target in entry['protoform'].items():
                    vocab.update(target)
                for lang, source in entry['daughters'].items():
                    vocab.update(source)
                dataset.append((cognate, entry))

        return dataset, vocab, langs

    @classmethod
    def get_cognateset_batch(cls, dataset, langs, C2I, L2I, device):
        """
        Convert both the daughter and protoform character lists to indices in the vocab
        """
        C2I = C2I._v2i
        cognatesets = {}

        protolang = langs[0]
        for cognate, entry in dataset:
            # L is the length of the cognate set - number of tokens in the set, including separators
            # lang_tensor specifies the language for each token in the input - (L,)
            # input_tensor specifies the input tensor - (L,)
            # target_tensor specifies the target tensor - (T,)

            # 1. convert the chars to indices
            # 2. then zip with the lang - List of (lang, index tensor)
            # 3. in the Embedding layer, add the BOS/EOS and the separator embeddings and do the language and char embeddings
            # the languages are supplied so the model knows what language embedding to apply
            target_tokens = []
            target_langs = []
            for lang, char in [('sep', "<")] + \
                [(protolang, char) for char in entry["protoform"][protolang]] + \
                [('sep', ">")]:
                target_tokens.append(C2I[char if char in C2I else "<unk>"])
                target_langs.append(L2I[lang])

            # print([I2C[idx] for idx in target_tokens])
            # print([(langs + ['sep'])[idx] for idx in target_langs])

            # TODO: don't do the to(device) here - move out to main.py
            target_tokens = torch.tensor(target_tokens).to(device)
            target_langs = torch.tensor(target_langs).to(device)

            # example cognate set (as a string)
            #     input: <*French:croître*Italian:crescere*Spanish:crecer*Portuguese:crecer*Romanian:crește*>
            #     protoform: <crescere>
            # note that the languages will be treated as one token

            # start of sequence
            source_tokens = [C2I["<"]]
            source_langs = [L2I['sep']]
            for lang in langs[1:]:
                source_token_sequence = [C2I['*'], C2I[lang], C2I[':']]
                # C2I treats the language tag as one token
                source_lang_sequence = [L2I['sep']] * 3
                # incomplete cognate set
                if lang not in entry['daughters']:
                    source_token_sequence.append(C2I['-'])
                    source_lang_sequence.append(L2I[lang])
                else:
                    raw_source_sequence = entry['daughters'][lang]
                    # note: C2I will recognize each language's name as a token, so it will not go to UNK
                    for char in raw_source_sequence:
                        source_token_sequence.append(C2I[char if char in C2I else "<unk>"])
                        source_lang_sequence.append(L2I[lang])
                source_tokens += source_token_sequence
                source_langs += source_lang_sequence
            # end of sequence
            source_tokens += [C2I["*"], C2I[">"]]
            source_langs += [L2I['sep'], L2I['sep']]

            # print(''.join([I2C[idx] for idx in source_tokens]))
            # print([(langs + ['sep'])[idx] for idx in source_langs])

            source_tokens = torch.tensor(source_tokens).to(device)
            source_langs = torch.tensor(source_langs).to(device)
            # source_tokens: (L,)
            # source_langs: (L,)
            # target_tokens: (T,)
            # target_langs: (T,)

            # used when calculating the loss
            cognatesets[cognate] = (source_tokens, source_langs, target_tokens, target_langs)

        return cognatesets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='chinese/romance_orthographic/romance_phonetic/austronesian')
    args = parser.parse_args()

    d = DataHandler(args.dataset)
    d.generate_split_datasets()
