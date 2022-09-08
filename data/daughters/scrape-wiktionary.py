import epitran
import cbor2
import re
import pandas as pd
import json


# to understand how Wiktionary's annotation scheme, refer to https://en.wiktionary.org/wiki/Template:zh-pron

# source: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "]+"
)
langs = ['Cantonese', 'Gan', 'Hakka', 'Jin', 'Mandarin', 'Hokkien', 'Wu', 'Xiang']
lang2code = {
    'Cantonese': 'c',
    'Gan': 'g',
    'Hakka': 'h',
    'Jin': 'j',
    'Mandarin': 'm',
    'Hokkien': 'mn',
    'Wu': 'w',
    'Xiang': 'x'
}
code2lang = { code:lang for lang, code in lang2code.items() }
variety_codes = code2lang.keys()
lang2epitrancode = {
    'Mandarin': 'cmn-Latn',
    'Jin': 'cjy-Latn',
    'Wu': 'wuu-Latn',
    'Gan': 'gan-Latn',
    'Cantonese': 'yue-Latn',
    'Hakka': 'hak-Latn',
    'Hokkien': 'nan-Latn',
    'Xiang': 'hsn-Latn',
}
transliterators = {}
for lang in langs:
    transliterators[lang] = epitran.Epitran(lang2epitrancode[lang], tones=True)

entry_counts = { variety:0 for variety in lang2code.keys()}
entry_counts['Middle Chinese'] = 0
# {'w': 9900, 'h': 18097, 'c': 94981, 'mn': 46634, 'g': 4317, 'x': 5076, 'j': 4944, 'm': 126641}
# {'w': 5155, 'h': 5311, 'c': 17134, 'mn': 7466, 'g': 2818, 'x': 2846, 'j': 2955, 'm': 19788} - single character

mandarin_variants = {}
with open('mandarin-variants.json') as f:
    mandarin_variants = json.load(f)
mc_heteronyms = set()


# transcriptions from Baxter for every entry in the Qieyun
baxter = pd.read_csv('../MC/mc-pron-baxter.csv', header=None, names=['Character', 'ID', 'Qieyun entry', 'Baxter transcription', 'IPA'])
baxter = baxter[['Character', 'Baxter transcription', 'IPA']]

baxter_heteronyms = pd.read_csv('../MC/mc-pron-baxter_heteronyms.csv', header=None, names=['Character', 'ID', 'Qieyun entry', 'Baxter transcription', 'IPA'])
baxter_heteronyms = baxter_heteronyms[['Character', 'Baxter transcription', 'IPA']]


def is_lettered(text):
    return re.findall(r'[A-Za-z0-9Α-Ωα-ω]', text)


def is_han(text):
    # TODO: need to be able to recognize CJK unicode extensions, e.g. 𪜶
    # alternative: not is_lettered()
    return re.findall(r'[\u4e00-\u9fff]+', text)


def parse_hakka(pron_entry):
    pfs, guangdong = (None, None)
    if len(pron_entry.split(';')) > 1:
        # both PFS (N Sixian) and Guangdong Romanization (Meixian) available
        pfs, guangdong = pron_entry.split(';')
        pfs, guangdong = pfs.split('pfs=')[1], guangdong.split(('gd=' if 'gd' in guangdong else 'gu='))[1]
    # only 1 pronunciation available
    elif "pfs" in pron_entry:
        pfs = pron_entry.split('pfs=')[1]
    elif "gd" in pron_entry:
        guangdong = pron_entry.split('gd=')[1]
    else:
        # malformatted entry
        return []

    if pfs:
        # remove Southern Sixian/Nansixian for now
        # keep all Northern Sixian pronunciations (either has no mark, has an n: mark, or has ns:)
        pfs_variants = pfs.split('/')
        pfs_north_sixian = list(filter(lambda pron: 'n:' in pron or 'ns:' in pron or (not ':' in pron), pfs_variants))

        def remove_marker(pron):
            # remove the ns - it means do not generate Southern Sixian
            if "ns:" in pron or "n:" in pron:
                return pron.split(':')[1]
            else:
                return pron
        pfs_north_sixian = list(map(remove_marker, pfs_north_sixian))

        # remove commas
        for i, variant in enumerate(pfs_north_sixian.copy()):
            if ',' in variant:
                comma_removed = variant.split(',')
                pfs_north_sixian[i] = comma_removed[0]
                pfs_north_sixian += comma_removed[1:]

        return pfs_north_sixian
    else:
        return []


def parse_hokkien(pron):
    # remove locations
    hok_variants = pron.split('/')

    def remove_marker(p):
        # remove the locations, which occur before the colon
        if ":" in p:
            return p.split(':')[1]
        else:
            return p

    return list(map(remove_marker, hok_variants))


def parse_mandarin(pron):
    final_entries = []
    mand_variants = pron.split(',')

    for variant in mand_variants:
        if "=" in variant or ";" in variant or "variant" in variant:
            # these are extra parameters like do/don't generate erhua
            # we do not need to worry about tone sandhi because we are dealing with single character words
            continue
        elif len(variant) > 0:
            # avoid empty string
            final_entries.append(variant)

    # if the Mandarin pronunciation entry is a Han character, Wiktionary will pull its pronunciation variants
    # from https://en.wiktionary.org/wiki/Module:zh/data/cmn-tag
    # we have adapted it into a JSON file (mandarin-variants.json)
    if len(final_entries) == 1 and is_han(final_entries[0]):
        if final_entries[0] in mandarin_variants:
            final_entries = mandarin_variants[final_entries[0]]
    elif len(final_entries) > 1 and is_han(final_entries[0]):
        # only cases: 圳, 頜. the Han character is the first element of the array
        # replace with Mandarin entries
        final_entries = mandarin_variants[final_entries[0]] + [final_entries[1]]

    return final_entries


def parse_middle_chinese(char, entry):
    baxter_transcriptions, ipa = ([], [])
    if entry == 'y':
        # displays all readings - fetch from mc-pron-baxter or the heteronyms file
        if baxter_heteronyms.loc[baxter_heteronyms['Character'].str.contains(char)].empty:
            # it's not a heteronym, fetch its one reading from mc-pron-baxter
            baxter_entry = baxter[baxter['Character'] == char]
            baxter_transcriptions += baxter_entry['Baxter transcription'].tolist()
            ipa += baxter_entry['IPA'].tolist()
        else:
            # if it's in the heteronym file, don't fetch from mc-pron-baxter
            baxter_entries = baxter_heteronyms.loc[baxter_heteronyms['Character'].str.contains(char)]
            baxter_transcriptions += baxter_entries['Baxter transcription'].tolist()
            ipa += baxter_entries['IPA'].tolist()
    elif '+' in entry:
        # displays corresponding readings
        # ex: the entry for 冷
        entry_indices = entry.split('+')
        for idx in entry_indices:
            baxter_entry = baxter_heteronyms[baxter_heteronyms['Character'] == char + '_' + str(int(idx) - 1)]
            baxter_transcriptions += baxter_entry['Baxter transcription'].tolist()
            ipa += baxter_entry['IPA'].tolist()
        mc_heteronyms.add(char)
    elif ',' in entry:
        # displays corresponding readings
        # the only characters that fall under this case: 礊,蝹,辟
        entry_indices = entry.split(',')
        for idx in entry_indices:
            baxter_entry = baxter_heteronyms[baxter_heteronyms['Character'] == char + '_' + str(int(idx) - 1)]
            baxter_transcriptions += baxter_entry['Baxter transcription'].tolist()
            ipa += baxter_entry['IPA'].tolist()
    elif entry.isnumeric():
        # the numbers are 1-indexed, while the heteronyms are zero indexed
        baxter_entry = baxter_heteronyms[baxter_heteronyms['Character'] == char + '_' + str(int(entry) - 1)]
        if baxter_entry.empty:
            # should only have 1 entry
            baxter_entry = baxter[baxter['Character'] == char]
        baxter_transcriptions += baxter_entry['Baxter transcription'].tolist()
        ipa += baxter_entry['IPA'].tolist()
    return baxter_transcriptions, ipa


cognates_romanization = {}
cognates_ipa = {}

count_complete = 0
near_complete = 0

# adapted from https://github.com/rime/rime-cantonese/blob/build/enwiktionary.py
# to obtain an updated zh-pron snapshot from Wiktionary, visit https://tools-static.wmflabs.org/templatehoard/dump/latest/zh-pron.cbor
cbor_path = 'zh-pron.cbor'
with open(cbor_path, 'rb') as file:
    decoder = cbor2.CBORDecoder(file)
    while file.peek(1):
        entry = decoder.decode()
        title = entry['title']

        # [[水/derived terms]]
        title = title.replace('/derived terms', '')

        # restrict to Han characters and words with only 1 character (no polysyllabic compounds)
        # also remove emojis
        if is_lettered(title) or len(title) > 1 or re.match(EMOJI_PATTERN, title):
            continue
        # 141,067 entries

        # if hanzidentifier.is_simplified(title) and not hanzidentifier.is_traditional(title):
        #     print(title, entry)

        # each character is unique within the CBOR snapshot
        character = title
        cognates_romanization[character] = []
        cognates_ipa[character] = []

        # multiple templates for an entry means there are multiple "Pronunciation" sections on Wiktionary
        # pronunciations across varieties believed to be cognate with each other are grouped together under a template
        for template in entry['templates']:
            """
            example of a template
            {'name': 'zh-pron', 'parameters': {'c': 'hyun2', 'c-t': 'hun2', 'cat': 'n', 'dg': '', 'g': 'qyon3', 'h': 'pfs=khién;gd=kian3', 'j': 'qye1', 'm': 'quǎn', 'm-s': 'quan3', 'mb': 'kṳǐng', 'mc': 'y', 'md': 'kēng', 'mn': 'ml,jj,tw:khián', 'mn-t': 'kiêng2/kiang2', 'mn-t_note': 'kiêng2 - Chaozhou; kiang2 - Shantou', 'oc': 'y', 'w': '2qyoe', 'x': 'qye3'}, 'text': None}
            """
            pronunciation_group_romanization = {}
            pronunciation_group_ipa = {}

            for lang_code, pron in template['parameters'].items():
                if len(pron) >= 1:
                    if lang_code not in variety_codes and lang_code != 'mc':
                        # do not have Epitran support yet for the language, so skip for now
                        continue

                    # remove HTML comments and replace with a -. these entries were commented out by users and should be left as blank
                    pron = re.sub(r'<!--(.*)-->', '', pron)
                    # remove tone after tone sandhi - preserve the base tone
                    pron = re.sub(r'(\d)-(\d)', r'\1', pron)
                    # remove space after comma
                    pron = re.sub(', ', ',', pron)
                    pron = pron.strip()
                    if len(pron) == 0:
                        pron = '-'

                    # before passing into epitran; separate the alternate pronunciations with spaces
                    if lang_code == "m":
                        pronunciations = parse_mandarin(pron)
                    elif lang_code == "c":
                        pronunciations = pron.split(',')
                    elif lang_code == "g":
                        pronunciations = pron.split('/')
                    elif lang_code == "h":
                        if parse_hakka(pron):
                            pronunciations = parse_hakka(pron)
                        else:
                            continue
                    elif lang_code == "j":
                        pronunciations = pron.split('/')
                    elif lang_code == "mn":
                        pronunciations = parse_hokkien(pron)
                    elif lang_code == "w":
                        pronunciations = pron.split(',')
                    elif lang_code == "x":
                        pronunciations = pron.split('/')
                    elif lang_code == "mc":
                        baxter_transcription, ipa = parse_middle_chinese(character, pron)
                        pronunciation_group_romanization['Middle Chinese (Baxter and Sagart 2014)'] = baxter_transcription
                        pronunciation_group_ipa['Middle Chinese (Baxter and Sagart 2014)'] = ipa
                        entry_counts["Middle Chinese"] += 1

                        # skip the Epitran transliteration
                        continue

                    lang = code2lang[lang_code]
                    entry_counts[lang] += 1

                    pronunciation_group_romanization[lang] = pronunciations
                    # if lang not in cognates_romanization[character]:
                    #     cognates_romanization[character][lang] = pronunciations
                    # else:
                    #     # do not overwrite the entries from a previous template ("Pronunciation" entry on Wiktionary)
                    #     cognates_romanization[character][lang] += pronunciations

                    transliterator = transliterators[lang]
                    # if lang not in cognates_ipa[character]:
                    #     cognates_ipa[character][lang] = [transliterator.transliterate(p) for p in pronunciations]
                    # else:
                    #     cognates_ipa[character][lang] += [transliterator.transliterate(p) for p in pronunciations]

                    pronunciation_group_ipa[lang] = [transliterator.transliterate(p) for p in pronunciations]

            cognates_romanization[character].append(pronunciation_group_romanization)
            cognates_ipa[character].append(pronunciation_group_ipa)


# make sure that the protolang comes first
langs = ['Middle Chinese (Baxter and Sagart 2014)'] + langs
heteronym_count = set()


romanization_output = 'wikihan-romanization.tsv'
ipa_output = 'wikihan-ipa.tsv'
with open(romanization_output, 'w') as rom_f, open(ipa_output, "w") as ipa_f, open('wikihan-ipa-recon.tsv', 'w') as temp_f:
    rom_f.write('Character\t' + '\t'.join(langs) + '\n')
    ipa_f.write('Character\t' + '\t'.join(langs) + '\n')
    temp_f.write('Character\t' + '\t'.join(langs) + '\n')
    for char in cognates_romanization:
        romanizations = cognates_romanization[char]
        transcriptions = cognates_ipa[char]

        if len(romanizations) > 1:
            heteronym_count.add(char)

            if any('Middle Chinese (Baxter and Sagart 2014)' in r for r in romanizations):
                mc_heteronyms.add(char)

        # iterate through each pronunciation group
        for romanizations, transcriptions in zip(romanizations, transcriptions):
            # use - to denote incomplete cognate set
            # / to join multiple variants of the same cognate within a language
            entries = ['/'.join(romanizations[lang]) if lang in romanizations else '-' for lang in langs]


            for i in range(len(entries)):
                if len(entries[i].strip()) == 0:
                    entries[i] = '-'
            # at least one entry in the 8 varieties we chose
            # some characters may have pronunciations in varieties we did not choose (e.g. Min Dong)
            if sum([1 for entry in entries if entry != '-']) > 0:
                rom_f.write(char + '\t')
                ipa_f.write(char + '\t')

                rom_f.write('\t'.join(entries))
                rom_f.write('\n')
            else:
                continue

            # TODO: create the JSON file here!

            ipa_entries = ['/'.join(transcriptions[lang]) if lang in transcriptions else '-' for lang in langs]
            for i in range(len(ipa_entries)):
                if len(ipa_entries[i].strip()) == 0:
                    ipa_entries[i] = '-'
            if sum([1 for entry in entries if entry != '-']) > 0:
                ipa_f.write('\t'.join(ipa_entries))
                ipa_f.write('\n')

            if 'Middle Chinese (Baxter and Sagart 2014)' in romanizations and ipa_entries[langs.index('Middle Chinese (Baxter and Sagart 2014)')] != '-' and len(list(filter(lambda entry: entry != '-', entries))) >= 4:
                temp_f.write(char + '\t')
                temp_f.write('\t'.join(ipa_entries))
                temp_f.write('\n')

        count_complete += (len(langs) == len(list(filter(lambda entry: entry != '-', entries))))
        near_complete += (abs(len(langs) - len(list(filter(lambda entry: entry != '-', entries)))) <= 2)

print("Pronunciation entries (not characters) per variety")
print(entry_counts)
print("Total pronunciation entries", sum(entry_counts.values()))
print(count_complete, "complete cognate sets")
print(near_complete, "near complete cognate sets (1 - 2 missing)")


with open('mc_heteronyms', 'w') as f:
    for char in mc_heteronyms:
        f.write(char + '\n')

with open('all-heteronyms', 'w') as f:
    for char in heteronym_count:
        f.write(char + '\n')
