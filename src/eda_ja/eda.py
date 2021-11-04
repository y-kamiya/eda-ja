from __future__ import annotations

import os
from dataclasses import dataclass
import re
import random
import MeCab
import sqlite3
import pandas as pd
from nltk.corpus import wordnet

from . import file_utils


@dataclass
class Words:
    raw: list[str]
    origin: list[str]

    def __post_init__(self):
        if not self.origin:
            return
        assert len(self.raw) == len(self.origin)

    def __len__(self):
        return len(self.raw)


class Eda():
    def __init__(self, stop_words_path):
        self.stop_words = []
        if self._has_stop_words(stop_words_path):
            with open(stop_words_path) as f:
                self.stop_words = [line.strip() for line in f.readlines()]

    def _has_stop_words(self, stop_words_path):
        if stop_words_path is None:
            return False

        if not os.path.isfile(stop_words_path):
            return False

        return True

    def synonym_replacement(self, words: Words, n: int):
        new_words = words.raw.copy()
        random_word_indexes = list(set([i for i, word in enumerate(words.raw) if word not in self.stop_words]))
        random.shuffle(random_word_indexes)

        num_replaced = 0
        for index in random_word_indexes:
            synonyms = self._get_synonyms(words.origin[index])

            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words[index] = synonym
                num_replaced += 1

            if num_replaced >= n:
                break

        return new_words

    def _get_synonyms(self, word):
        synsets = self.wordnet.loc[self.wordnet.lemma == word, "synset"]
        synset_words = set(self.wordnet.loc[self.wordnet.synset.isin(synsets), "lemma"])

        if word in synset_words:
            synset_words.remove(word)

        return list(synset_words)

    def random_deletion(self, words, p):

        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self._swap_word(new_words)
        return new_words

    def _swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self._add_word(new_words)
        return new_words

    def _add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self._get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def _concat_words(self, words):
        return ' '.join(words)

    def clean(self, line):
        line = line.replace("’", "")
        line = line.replace("'", "")
        line = line.replace("-", " ") #replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        return line.lower()

    def generate_sentences(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        sentence = self.clean(sentence)
        words = self._parse(sentence)
        num_words = len(words)
        
        augmented_sentences = []
        num_new_per_technique = int(num_aug/4)+1

        #sr
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sentences.append(self._concat_words(a_words))

        #ri
        if (alpha_ri > 0):
            n_ri = max(1, int(alpha_ri*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words.raw, n_ri)
                augmented_sentences.append(self._concat_words(a_words))

        #rs
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words.raw, n_rs)
                augmented_sentences.append(self._concat_words(a_words))

        #rd
        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words.raw, p_rd)
                augmented_sentences.append(self._concat_words(a_words))

        augmented_sentences = [self.clean(sentence) for sentence in augmented_sentences]
        random.shuffle(augmented_sentences)

        #trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        #append the original sentence
        augmented_sentences.append(sentence)

        return augmented_sentences


class EdaJa(Eda):
    URL_WORDNET = "http://compling.hss.ntu.edu.sg/wnja/data/1.1/wnjpn.db.gz"

    def __init__(self, stop_words_path, wordnet_path):
        super().__init__(stop_words_path)
        self.tagger = MeCab.Tagger()
        self.wordnet = self._create_wordnet(wordnet_path)

    def _create_wordnet(self, wordnet_path):
        path = file_utils.get_from_cache(wordnet_path, self.URL_WORDNET)
        conn = sqlite3.connect(path)
        query = 'SELECT synset,lemma FROM sense,word USING (wordid) WHERE sense.lang="jpn"'
        return pd.read_sql(query, conn)

    def _concat_words(self, words):
        return ''.join(words)

    def _parse(self, sentence) -> Words:
        node = self.tagger.parseToNode(sentence)

        words = []
        original_words = []
        while node:
            words.append(node.surface.lower())

            pos = node.feature.split(",")
            original_words.append(pos[6] if pos[0] in ["動詞", "名詞"] else "")

            node = node.next

        return Words(words, original_words)


class EdaEn(Eda):
    def _concat_words(self, words):
        return ' '.join(words)

    def _parse(self, sentence) -> Words:
        words = [word for word in sentence.split(' ') if word != '']
        return Words(words, [])

    def clean(self, line):
        line = super().clean(line)

        clean_line = ""

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def synonym_replacement(self, words: Words, n: int):
        new_words = words.raw.copy()
        random_word_list = list(set([word for word in words.raw if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1

            if num_replaced >= n:
                break

        #this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def _get_synonyms(self, word):
        try:
            synsets = wordnet.synsets(word)
        except LookupError:
            import nltk
            nltk.download('wordnet')
            synsets = wordnet.synsets(word)

        synonyms = set()
        for syn in synsets:
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
