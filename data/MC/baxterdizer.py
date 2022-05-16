#!/usr/bin/env python3

######################################################################
"""
baxterdizer.py

A program for generating Middle Chinese transcriptions after the manner
of Bill Baxter from the Middle Chinese descriptions given in the DOC.

Based on <redacted>'s Python2 implementation. Adapted for Python3
by <redacted>
"""
######################################################################

from baxter_mc import INITIALS, FINALS, TONES, FINAL_CONV
import pandas as pd
from collections import defaultdict
import epitran

DIVISIONS = {'一': 0,
             '二': 1,
             '三': 2,
             '四': 3}

unknown_initial = defaultdict(int)
unknown_final = defaultdict(int)

class Baxterdizer:
    def __init__(self, input_files=None):
        if input_files is None:
            input_files = ['mc-pron.csv']
        columns = ["hz", "id", "mc_desc"]
        self.data = pd.read_csv(input_files[0], names=columns)
        for i in range(1, len(input_files)):
            self.data = self.data.append(pd.read_csv(input_files[i], names=columns))

        print("translating from desecription to baxter")
        self.data['baxter_trans'] = self.data['mc_desc'].apply(self.desc_to_baxter)
        # print("unknown initial")
        # print(unknown_initial)
        # print("unknown final")
        # print(unknown_final)
        print("translating from baxter to ipa")

        # this mimics baxter2ipa in sinopy, but with more flexibility
        self.epi = epitran.Epitran("ltc-Latn-bax")
        self.data['baxter_ipa'] = self.data['baxter_trans'].apply(self.baxter_to_ipa)

    def baxter_to_ipa(self, trans):
        trans = self.epi.transliterate(trans)
        # add ligatures for affricates
        trans = trans.replace("ts", "t͡s").replace("dz", "d͡z")
        trans = trans.replace("tɕ", "t͡ɕ").replace("dʑ", "d͡ʑ")
        trans = trans.replace("ʈʂ", "ʈ͡ʂ").replace("ɖʐ", "ɖ͡ʐ")
        return trans

    def desc_to_baxter(self, desc):
        if len(desc) < 6:
            print("cannot parse mc description:", desc)
            return "(FAILED)"
        initial = self.make_initial(desc[0])
        final = self.make_final(desc[1], desc[2], desc[3], desc[5])
        tone = self.make_tone(desc[5])
        return self.regularize_trans(initial + final + tone)

    def make_initial(self, initial):
        try:
            return INITIALS[initial]
        except KeyError:
            if initial not in unknown_initial:
                print("unknown initial:", initial)
            unknown_initial[initial] += 1
            return "(FAILED)"

    def make_final(self, final, div, hekai, tone):
        if tone == '入':
            final = FINAL_CONV[final]
        try:
            tr = FINALS[final][DIVISIONS[div]]
        except KeyError:
            if final not in unknown_final:
                print("unknown final:", final)
            unknown_final[final] += 1
            return "(FAILED)"
        if tr == "":
            print("invalid division for final:", final, div)

        if hekai == '合':
            tr = tr.replace('(w)', 'w')
        else:  # 開
            tr = tr.replace('(w)', '')
        return tr

    def make_tone(self, tone):
        try:
            return TONES[tone]
        except KeyError:
            print("unknown tone:", tone)
            return "(FAILED)"

    def regularize_trans(self, trans):
        if 'FAILED' in trans:
            return '-'

        trans = trans.replace('yj', 'y')
        trans = trans.replace('jj', 'j')
        trans = trans.replace('+', 'ɨ')
        trans = trans.replace('ea', 'ɛ')
        trans = trans.replace('ae', 'æ')
        trans = trans.replace("'", 'ʔ')

        if trans[0] in ('p', 'b', 'm'):
            trans = trans.replace('(i)', 'i')
        else:
            trans = trans.replace('(i)', '')

        return trans

    def write_data_csv(self, out_dir):
        self.data.to_csv(out_dir, header=False, index=False)


app = Baxterdizer()
app.write_data_csv("mc-pron-baxter.csv")

# app = Baxterdizer(input_files=["mc-pron-heteronyms.csv"])
# app.write_data_csv("mc-pron-baxter_heteronyms.csv")
