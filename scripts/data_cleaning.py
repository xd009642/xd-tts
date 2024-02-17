from gruut import sentences

bad_phones = set()

def convert(ipa):
    if ipa.startswith("'") or ipa.startswith("ˈ"):
        ipa = ipa[1:]
        aux = "1 "
    elif ipa.startswith("ˌ"):
        ipa=ipa[1:] 
        aux = "2 "
    else:
        aux =" "

    if ipa in [ "ɒ" , "ɑ"]:
        phone = "AA"
    elif ipa == "æ":
        phone = "AE"
    elif ipa in ["ʌ", "ə"]:
        phone = "AH"
    elif ipa == "ɔ":
        phone = "AO"
    elif ipa == "aʊ":
        phone = "AW"
    elif ipa == "aɪ":
        phone = "AY"
    elif ipa == "ɛ":
        phone = "EH"
    elif ipa in ["ɝ", "ɚ"]:
        phone = "ER"
    elif ipa == "eɪ":
        phone = "EY"
    elif ipa == "ɪ":
        phone = "IH"
    elif ipa == "i":
        phone = "IY"
    elif ipa == "oʊ":
        phone = "OW"
    elif ipa == "ɔɪ":
        phone = "OY"
    elif ipa == "ʊ":
        phone = "UH"
    elif ipa == "u":
        phone = "UW"
    elif ipa == "b":
        phone = "B"
    elif ipa in ["tʃ", "t͡ʃ"]:
        phone = "CH"
    elif ipa == "d":
        phone = "D"
    elif ipa == "ð":
        phone = "DH"
    elif ipa == "f":
        phone = "F"
    elif ipa == "ɡ":
        phone = "G"
    elif ipa == "h":
        phone = "HH"
    elif ipa in  ["dʒ", "d͡ʒ"]:
        phone = "JH"
    elif ipa == "k":
        phone = "K"
    elif ipa == "l":
        phone = "L"
    elif ipa == "m":
        phone = "M"
    elif ipa == "n":
        phone = "N"
    elif ipa == "ŋ":
        phone = "NG"
    elif ipa == "p":
        phone = "P"
    elif ipa == "ɹ":
        phone = "R"
    elif ipa == "s":
        phone = "S"
    elif ipa == "ʃ":
        phone = "SH"
    elif ipa == "t":
        phone = "T"
    elif ipa == "θ":
        phone = "TH"
    elif ipa == "v":
        phone = "V"
    elif ipa == "w":
        phone = "W"
    elif ipa == "j":
        phone = "Y"
    elif ipa == "z":
        phone = "Z"
    elif ipa == "ʒ":
        phone = "ZH"
    else:
        bad_phones.add(ipa)
        return "<UNK> "

    return f"{phone}{aux}"

words = set()
vocab = dict()

with open('logs.txt') as fd:
    lines = fd.readlines()

    for line in lines:
        s = line.split("'")[-2].strip()
        if len(s) > 0:
            words.add(s)

with open('librispeech-lexicon.txt') as d:
    for line in d.readlines():
        s = line.split('  ')
        if len(s) == 1:
            s = line.split('\t')
        vocab[s[0]] = s[1].strip()

still_oov = set()
with open('words.txt', 'w') as fd:
    for word in words:
        if word in vocab:
            fd.write('{}  {}\n'.format(word, vocab[word]))
        else: 
            print(f"Didn't handle: {word}")
            still_oov.add(word)

print(f"Unhandled words: {len(still_oov)}")

dont_recase = ["UV", "MVD", "LLD", "NRA", "MPS", "WDSU", "MWDDY", "VC", "LJ", "BBL"]

with open('unhandled.txt', 'w') as fd:
    for word in still_oov:
        string=""
        i = 0

        if not word in dont_recase:
            word = word.lower()

        for sent in sentences(word, lang="en-us"):
            for w in sent:
                arpa_phonemes = map(convert, w.phonemes)
                string = string + "".join(arpa_phonemes)
                i = i + 1
        if i > 1:
            print(f'{word} had multiple passes!?')

        if len(string) > 0:
            fd.write(f'{word.upper()}  {string.strip()}\n')

print(f"unique {len(bad_phones)} unrecognised phonemes")
print(", ".join(bad_phones))
