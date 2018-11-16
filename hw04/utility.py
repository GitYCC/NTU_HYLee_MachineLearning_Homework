import re
import math
import string
import pickle

import numpy as np


STOPWORDS = [
    'a', 'about', 'above', 'above', 'across', 'after', 'afterwards',
    'again', 'against', 'all', 'almost', 'alone', 'along',
    'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
    'amount',  'an', 'and', 'another', 'any', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'are', 'around', 'as',  'at', 'back',
    'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been',
    'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
    'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call',
    'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
    'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each',
    'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough',
    'etc', 'even', 'ever', 'every', 'everyone', 'everything',
    'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire',
    'first', 'five', 'for', 'former', 'formerly', 'forty', 'found',
    'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has',
    'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter',
    'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed',
    'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter',
    'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me',
    'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
    'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
    'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none',
    'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off',
    'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others',
    'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part',
    'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem',
    'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should',
    'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow',
    'someone', 'something', 'sometime', 'sometimes', 'somewhere',
    'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
    'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
    'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv',
    'thin', 'third', 'this', 'those', 'though', 'three', 'through',
    'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
    'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up',
    'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever',
    'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
    'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why',
    'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'the'
]


class BoW(object):
    def __init__(self):
        self.__words = set()
        self.__numbers = {}

    def _add_word(self, word):
        if word == '':
            return
        if word not in self.__words:
            self.__numbers[word] = 0
            self.__words.add(word)

        self.__numbers[word] += 1

    def _add_line(self, line):
        for word in re.split(' +', line.strip()):
            self._add_word(word)

    def add_from_lines(self, lines):
        for line in lines:
            self._add_line(line)

    def add_from_text(self, text):
        for line in text.split('\n'):
            self._add_line(line)

    def __add__(self, that):
        """Concat bow."""
        words = self._words.union(that._words)
        numbers = {}
        for word in list(words):
            num1 = self.__numbers.get(word) if self.__numbers.get(word) else 0
            num2 = that.__numbers.get(word) if that.__numbers.get(word) else 0
            numbers[word] = num1 + num2

        new_BoW = BoW()
        new_BoW.__words = words
        new_BoW.__numbers = numbers
        return new_BoW

    def todict(self):
        return self.__numbers

    @property
    def words(self):
        return list(self.__words)

    def count(self):
        return sum([val for _, val in self.__numbers.items()])

    def freq_dict(self):
        count = self.count()
        dict_ = {}
        for key, val in self.__numbers.items():
            dict_[key] = val / count
        return dict_


class Corpus(object):
    def __init__(self):
        self.__docs_name = []
        self.__docs_bow = []
        self.__word_vector = []

    def dump(self, file):
        dict_ = {
            'docs_name': self.__docs_name,
            'docs_bow': self.__docs_bow,
            'word_vector': self.__word_vector
        }
        pickle.dump(dict_, open(file, 'wb'))

    def load(self, file):
        dict_ = pickle.load(open(file, 'rb'))
        self.__docs_name = dict_['docs_name']
        self.__docs_bow = dict_['docs_bow']
        self.__word_vector = dict_['word_vector']

    def add_doc_from_bow(self, bow, name=None):
        self.__docs_name.append(name if name else 'doc{:06}'.format(len(self.__docs_name)+1))
        self.__docs_bow.append(bow)
        for word in bow.words:
            if word not in self.__word_vector:
                self.__word_vector.append(word)

    def add_docs_from_bows(self, bows, names=None):
        if names:
            for bow, name in zip(bows, names):
                self.add_doc_from_bow(bow, name)
        else:
            for bow in bows:
                self.add_doc_from_bow(bow)

    def add_doc_from_text(self, text, name=None):
        bow = BoW()
        bow.add_from_text(text)
        self.add_doc_from_bow(bow, name)

    def num_docs(self):
        return len(self.__docs_bow)

    @property
    def word_vector(self):
        self.__word_vector = sorted(self.__word_vector)
        return np.array(self.__word_vector)

    @property
    def doc_vector(self):
        return np.array(self.__docs_name)

    @property
    def docs_bow(self):
        return self.__docs_bow

    def to_doc_word_matrix(self):
        matrix = np.zeros((len(self.__docs_bow), self.word_vector.shape[0]))
        list_word_vector = self.word_vector.tolist()
        for i, bow in enumerate(self.__docs_bow):
            for word, num in bow.items():
                matrix[i, list_word_vector.index(word)] = num

        return matrix


class TFIDF(object):
    def __init__(self, corpus):
        self.__corpus = corpus

    def get_term_frequency(self):
        """Calcuate Term Frequency (TF)."""
        tf = [bow.freq_dict() for bow in self.__corpus.docs_bow]
        return tf

    def get_inverse_document_frequency(self):
        """Calcuate Inverse Document Frequency (IDF)."""
        D = self.__corpus.num_docs()
        docs_words = [bow.words for bow in self.__corpus.docs_bow]

        tmp_idf = {}
        for ws in docs_words:
            for w in ws:
                if w not in tmp_idf.keys():
                    tmp_idf[w] = 0
                tmp_idf[w] += 1
        idf = {}
        for key, val in tmp_idf.items():
            idf[key] = math.log(float(D)/float(val))

        return idf

    def get_tfidf(self):
        print('Get Term Frequenct ...')
        tf = self.get_term_frequency()
        print('Get Term Frequenct ... Finished')
        print('Get Inverse Document Frequency ...')
        idf = self.get_inverse_document_frequency()
        print('Get Inverse Document Frequency ... Finished')
        print('Generate TF-IDF Dict ...')
        tfidf = [{key: val*idf[key] for key, val in doc_tf.items()} for doc_tf in tf]
        print('Generate TF-IDF Dict ... Finished')
        return tfidf

    def get_tfidf_matrix(self):
        tfidf = self.get_tfidf()
        print('Start Generating Matrix')
        matrix = np.zeros((len(tfidf), self.__corpus.word_vector.shape[0]))
        list_word_vector = self.__corpus.word_vector.tolist()
        for i, dict_ in enumerate(tfidf):
            print ('Generate Matrix: {}/{}'.format(i+1, len(tfidf)))
            for word, weight in dict_.items():
                matrix[i, list_word_vector.index(word)] = weight
        return matrix


class TextProcess(object):
    @staticmethod
    def shrink_whitespace(text):
        return re.sub(r'[ \t]+', ' ', text, flags=re.MULTILINE)

    @staticmethod
    def remove_url(text):
        return re.sub(r'https?:\/\/[^\s]*([ \r\n])', ' \\1', text, flags=re.MULTILINE)

    @staticmethod
    def remove_html(text):
        return re.sub(r'<[^>\n]+>', '', text, flags=re.MULTILINE)

    @staticmethod
    def remove_stopword(text):
        ntext = ''
        for line in text.split('\n'):
            nline = []
            for word in line.strip().split(' '):
                if word not in STOPWORDS:
                    nline.append(word)
            ntext += ' '.join(nline) + '\n'

        return ntext

    @staticmethod
    def tolower(text):
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        return text.translate(table)

    @staticmethod
    def remove_number(text):
        return re.sub('[0-9]', '', text, flags=re.MULTILINE)

    @staticmethod
    def shrink_empty_line(text):
        return '\n'.join(map(lambda line: re.sub(r'^ +$', '', line), text.split('\n')))


def test_tfidf():
    corpus = Corpus()
    for doc in ['apple banana class', 'class run run', 'class under apple']:
        corpus.add_doc_from_text(doc)
    ycTFIDF = TFIDF(corpus)
    word_vector = corpus.word_vector
    matrix = ycTFIDF.get_tfidf_matrix()
    print(word_vector)
    print(matrix)


if __name__ == '__main__':
    test_tfidf()
