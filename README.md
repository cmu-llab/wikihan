# WikiHan
A comparative Sinitic dataset drawn from Wiktionary that can be used for computational Middle Chinese reconstruction and more. 
For more details, please refer to Chang et al 2022. 


# Statistics
Total entries: 67,943

Cognate sets: 21,227

Varieties: 8 (not including Middle Chinese)

| Subgroup         | Dialect Chosen | Pronunciation entries |
|------------------|----------------|-----------------------|
| Mandarin         | Beijing        | 20369                 |
| Yue              | Cantonese      |  16727                |
| Min              | Hokkien*       | 6185                  |
| Hakka            | Sixian         | 3269 |
| Wu               | Shanghainese   | 2877 |
| Jin              | Taiyuan        | 1410 |
| Xiang            | Old Xiang      | 1258 |
| Gan              | Nanchang       | 1195 |
| ---------------  | -------------- |------|
| Middle Chinese** | Baxter and Sagart (2014) | 14653 |

A pronunciation entry is one cell in the TSV. It can contain 1 pronunciation or many pronunciation variants. 
A cognate set is one row in our table. A heteronymic character can have multiple cognate sets, reflecting different sets of pronunciation variants that are only cognate with the variants in the same set. 

*For Min, the pronunciations are a mix of the Xiamen, Quanzhou, Zhangzhou, and Taiwanese dialects of Hokkien. 
**Middle Chinese is not a subgroup, but we include it in the table for convenience.


# Files

```
├── LICENSE: Creative Commons License
├── data
|    ├── MC
|    |   ├── baxter_mc.py: mappings going from Qieyun to Baxter and Sagart's ASCII transcription
|    |   ├── baxterdizer.py: Script converting Baxter and Sagart's ASCII-based Middle Chinese transcription to IPA
|    |   ├── ltc-Latn-bax.csv: Epitran mapping table going from a Baxter transcription character to an IPA phoneme; not used in this repo but included for reference
|    |   ├── mc-pron-baxter.csv: mc-pron.csv with IPA pronunciations
|    |   ├── mc-pron-baxter_heteronyms.csv: mc-pron-baxter.csv but with Middle Chinese heteronyms
|    |   ├── mc-pron.csv: Qieyun (Middle Chinese) entries from https://github.com/ycm/cs221-proj/blob/master/preprocessing/dataset/pron/mc-pron.csv
|    |   └── retrieve_mc_pron.py: an obsolete script to obtain Middle Chinese pronunciations from Wiktionary from the front end
|    └── daughters
|        ├── mandarin-variants.json: characters with pronunciation variants in Mandarin; adapted from Wiktionary
|        ├── scrape-wiktionary.py: the script we used to generate the dataset
|        └── zh-pron.cbor: snapshot of Wiktionary from 02-Sep-2022 21:01
├── requirements.txt: dependencies needed to run the scrape-wiktionary.py script
├── wikihan-ipa-reconstruction.tsv: Version of the dataset with at least 3 daughters (at least 4 entries including Middle Chinese)
├── wikihan-ipa.tsv: The dataset with pronunciations in International Phonetic Alphabet
└── wikihan-romanization.tsv: The dataset with pronunciations left in their romanization (the same form in the Wiktionary snapshot)
```

# Updating the data

To obtain an updated snapshot from Wiktionary, visit https://tools-static.wmflabs.org/templatehoard/dump/latest/zh-pron.cbor
and replace data/daughters/zh-pron.cbor. 

Then re-generate the data using our script:
```
pip install -r requirements.txt
cd data/daughters
python scrape-wiktionary.py
```


# Citing WikiHan

Please cite WikiHan as follows:

Kalvin Chang, Chenxuan Cui, Youngmin Kim, and David R. Mortensen. 2022. WikiHan: A New Comparative Dataset for Chinese Languages. In *Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022)*, Gyeongju, Korea.


```
@InProceedings{Chang-et-al:2022,
  author = {Chang, Kalvin and Cui, Chenxuan and Kim, Youngmin and Mortensen, David R.},
  title = {Wiki{H}an: {A} New Comparative Dataset for {C}hinese Languages},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022)},
  year = {2022},
  month = {October},
  date = {12--17},
  location = {Gyeongju, Korea},
}
```


