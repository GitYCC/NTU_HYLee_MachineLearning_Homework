import os
import re
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    BatchNormalization,
    # Flatten,
    # Conv2D,
    # MaxPooling2D,
    # UpSampling2D,
    Dense,
    Dropout,
    # Lambda,
)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Normalizer

from utility import (
    # BoW,
    Corpus,
    TFIDF,
    TextProcess,
)


ROOT = os.path.dirname(os.path.abspath(__file__))


class AutoEncoder(object):
    def __init__(self, inputs, file_load_weights=None):
        def norm_relu(in_layer):
            return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

        input_ = Input(shape=(inputs, ))

        encoded = Dense(80)(input_)
        encoded = norm_relu(encoded)
        encoded = Dropout(0.3)(encoded)

        encoded = Dense(10)(encoded)

        decoded = norm_relu(encoded)

        decoded = Dense(80)(decoded)
        decoded = norm_relu(decoded)
        decoded = Dropout(0.3)(decoded)

        decoded = Dense(inputs)(decoded)

        ae = Model(inputs=input_, outputs=decoded)
        ae.compile(loss='mean_squared_error', optimizer='adam')

        encoder = Model(inputs=input_, outputs=encoded)
        encoder.compile(loss='mean_squared_error', optimizer='adam')

        self.ae = ae
        self.encoder = encoder
        self.ae_batch = 64
        self.ae_epoch = 5
        self.ratio_validation = 0.1

    def train(self, X, model_path):  # noqa: N803
        self.ae.summary()

        self.ae.fit(
            X, X,
            batch_size=self.ae_batch,
            epochs=self.ae_epoch,
            validation_split=self.ratio_validation,
            callbacks=[
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                save_weights_only=True),
                EarlyStopping(monitor='val_loss', patience=3, mode='min')
                ])

    def extract_feature(self, X):  # noqa: N803
        # freeze pretrain layers
        for layer in self.ae.layers:
            layer.trainable = False

        self.encoder.summary()
        return self.encoder.predict(X, batch_size=32)


def text_preproc(text):
    text = TextProcess.shrink_whitespace(text)
    text = TextProcess.tolower(text)
    text = TextProcess.remove_html(text)
    text = TextProcess.remove_url(text)
    text = TextProcess.remove_number(text)
    text = TextProcess.remove_punctuation(text)
    text = TextProcess.remove_stopword(text)
    text = TextProcess.shrink_empty_line(text)
    return text


def kmeans_classify(features, n_clusters=20):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100, n_init=20)
    kmeans.fit(features)
    return kmeans.labels_


def map_word_vector(target_vector, vector, matrix):
    list_vector = vector.tolist()
    list_target = target_vector.tolist()
    index = [list_vector.index(word) if word in list_vector else -1 for word in list_target]
    num_data = matrix.shape[0]
    argu_matrix = np.hstack((matrix, np.array([[0] for _ in range(num_data)])))
    return argu_matrix[:, index]


def reduce_by_tfidf(word_vector, matrix):
    THRESHOLD_TFIDF = 2.0
    choose = np.max(matrix, axis=0) >= THRESHOLD_TFIDF

    return (word_vector[choose], matrix[:, choose])


def pca_visualization(features, class_, real_class=None, label=None):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    try:
        colors = mpl.colors.XKCD_COLORS
    except:
        colors = pickle.load(open(os.path.join(ROOT, 'colors.p'), 'rb'))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 20))

    data_predict = {}
    for (x, y), c in zip(reduced_features, class_):
        if not data_predict.get(c):
            data_predict[c] = [[], []]
        data_predict[c][0].append(x)
        data_predict[c][1].append(y)

    for c in data_predict:
        color = colors[list(colors.keys())[c*2]]
        xs = data_predict[c][0]
        ys = data_predict[c][1]
        ax1.scatter(xs, ys, color=color)

    if real_class:
        # mapping class number, using numbers in class
        ndarray_real_class = np.array(real_class)
        ndarray_class = np.array(class_)

        classes = sorted(np.unique(ndarray_real_class).tolist())
        num_classes = len(classes)
        overlay_matrix = np.empty((0, num_classes))

        for c in classes:
            group = ndarray_class[ndarray_real_class == c]
            overlay_count = np.bincount(group)
            if overlay_count.shape[0] != num_classes:
                overlay_count = np.append(
                    overlay_count, np.zeros(num_classes-overlay_count.shape[0]))
            overlay_matrix = np.append(overlay_matrix, [overlay_count], axis=0)

        mapped_index = {}
        while int(np.max(overlay_matrix)) >= 0:
            max_ = np.max(overlay_matrix)
            xs, ys = np.where(overlay_matrix == max_)
            for i in range(xs.shape[0]):
                x = xs[i]
                y = ys[i]
                if (mapped_index.get(x) is None) and (y not in list(mapped_index.values())):
                    mapped_index[x] = y
            overlay_matrix[overlay_matrix == max_] = -1

        data_real = {}
        for (x, y), old_c in zip(reduced_features, real_class):
            c = mapped_index[old_c]
            if not data_real.get(c):
                data_real[c] = [[], []]
            data_real[c][0].append(x)
            data_real[c][1].append(y)

        handles = []
        for c in data_real:
            color = colors[list(colors.keys())[c*2]]
            xs = data_real[c][0]
            ys = data_real[c][1]

            if label:
                handle = ax2.scatter(xs, ys, color=color, label=label[c])
                handles.append(handle)
            else:
                ax2.scatter(xs, ys, color=color)

        if label:
            plt.legend(handles=handles)
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print('Finish Ploting')
    plt.show()


def main():
    global test_word_vector, test_matrix, corpus, train_matrix, train_word_vector

    path_data_titles = os.path.join(ROOT, 'data', 'title_StackOverflow.txt')
    path_data_docs = os.path.join(ROOT, 'data', 'docs.txt')
    path_data_label = os.path.join(ROOT, 'data', 'label_StackOverflow.txt')
    path_preproc_docs = os.path.join(ROOT, 'tmp', 'preproc_docs.txt')
    path_corpus = os.path.join(ROOT, 'tmp', 'corpus.p')
    path_matrix = os.path.join(ROOT, 'tmp', 'matrix.npz')
    path_model_dir = os.path.join(ROOT, 'models')

    # ### document preprocess
    print('### Document Preprocess')
    if not os.path.exists(path_preproc_docs):
        fr = open(path_data_docs, 'r', encoding='utf8')
        docs = fr.read()
        docs = text_preproc(docs)

        fw = open(path_preproc_docs, 'w', encoding='utf8')
        fw.write(docs)
        fr.close()
        fw.close()

    preproc_docs = open(path_preproc_docs, 'r', encoding='utf8').read()

    # ### train corpus
    print('### Train Corpus')
    if not os.path.exists(path_corpus):
        corpus = Corpus()
        # take empty line as space of different file
        # for doc in re.split('\n\n+', preproc_docs):
        #    corpus.add_doc_from_text(doc)

        # take one line as different file
        doc_list = re.split('\n', preproc_docs)
        doc_list = list(filter(lambda x: x.strip() != '', doc_list))
        doc_list = random.sample(doc_list, int(len(doc_list)/2))
        for doc in doc_list:
            corpus.add_doc_from_text(doc)

        corpus.dump(path_corpus)

    corpus = Corpus()
    corpus.load(path_corpus)

    del preproc_docs

    # ### Matrix
    print('### Matrix')
    if not os.path.exists(path_matrix):
        # ## train matrix: docs
        print('## train matrix: docs')
        ycTFIDF = TFIDF(corpus)
        train_word_vector = corpus.word_vector
        train_matrix = ycTFIDF.get_tfidf_matrix()
        train_word_vector, train_matrix = reduce_by_tfidf(train_word_vector, train_matrix)

        del corpus

        # ## test matrix: stackoverflow title
        print('## test matrix: stackoverflow title')
        with open(path_data_titles, 'r', encoding='utf8') as fr:
            test_corpus = Corpus()
            for line in fr.readlines():
                line = text_preproc(line)
                test_corpus.add_doc_from_text(line)

            test_word_vector = test_corpus.word_vector
            test_matrix = TFIDF(test_corpus).get_tfidf_matrix()

        # ## common word vector
        print('## common word vector')
        list_train_word_vector = train_word_vector.tolist()
        list_test_word_vector = test_word_vector.tolist()
        common_word_vector = np.array(
            list(filter(lambda x: x in list_train_word_vector, list_test_word_vector)))

        # ## mapping
        print('## mapping')
        train_matrix_mapped = map_word_vector(
            target_vector=common_word_vector, vector=train_word_vector, matrix=train_matrix)
        del train_matrix
        del train_word_vector
        test_matrix_mapped = map_word_vector(
            target_vector=common_word_vector, vector=test_word_vector, matrix=test_matrix)
        del test_matrix
        del test_word_vector

        # ## save matrix
        print('## save matrix')
        np.savez(path_matrix,
                 train_matrix_mapped=train_matrix_mapped,
                 test_matrix_mapped=test_matrix_mapped,
                 common_word_vector=common_word_vector)

    npzfile = np.load(path_matrix)

    common_word_vector = npzfile['common_word_vector']
    train_matrix_mapped = npzfile['train_matrix_mapped']
    test_matrix_mapped = npzfile['test_matrix_mapped']

    del npzfile

    # load label data
    with open(path_data_label, 'r') as fr:
        test_label = []
        for line in fr.readlines():
            line = line.strip()
            num = int(line)
            # begin at 0
            num = num - 1
            test_label.append(num)

    LABEL = ['wordpress', 'oracle', 'svn', 'apache', 'excel', 'matlab', 'visual studio',
             'cocoa', 'osx', 'bash', 'spring', 'hibernate', 'scala', 'sharepoint', 'ajax',
             'qt', 'drupal', 'linq', 'haskell', 'magento']

    METHOD = 'autoencoder'

    if METHOD == 'PCA':
        print('### PCA')
        pca = PCA(n_components=2)
        pca.fit(train_matrix_mapped)
        print('pca finish')
        test_features = pca.transform(test_matrix_mapped)

    if METHOD == 'PCA_self':
        print('### PCA')
        pca = PCA(n_components=2)
        pca.fit(test_matrix_mapped)
        print('pca finish')
        test_features = pca.transform(test_matrix_mapped)

    elif METHOD == 'autoencoder':
        ae = AutoEncoder(inputs=train_matrix_mapped.shape[1], file_load_weights=None)
        ae.train(train_matrix_mapped, os.path.join(path_model_dir, 'ae.model'))
        test_features = ae.extract_feature(test_matrix_mapped)

    elif METHOD == 'SVD':
        pass
        # svd = TruncatedSVD(20)
        # normalizer = Normalizer(copy=False)
        # lsa = make_pipeline(svd, normalizer)

        # X = lsa.fit_transform(X)

    test_classify = kmeans_classify(features=test_features, n_clusters=20)
    print('kmeans finish')
    pca_visualization(
        features=test_features, class_=test_classify, real_class=test_label, label=LABEL)


if __name__ == '__main__':
    main()
