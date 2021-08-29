import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import classla
import os
import pickle
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import time

from tqdm import tqdm

tqdm.pandas()

import tensorflow as tf


def basic_info(df):
    # number of cerif codes
    print(df['cerif'].value_counts())

    get_unique_subcerif(df)

    # split data into cerif
    df_subcerif = df[df['cerif_conf'] == 'high']
    for label, g in df_subcerif.groupby('cerif'):
        print(label)
        get_unique_subcerif(g)
        print('\n' * 3)


# unique number of subcerif codes
def get_unique_subcerif(df):
    s = set()
    all = []
    for label in df['subcerif']:
        if '|' in label:
            for l in label.split('|'):
                s.add(l)
                all.append(l)
        else:
            s.add(label)
            all.append(label)

    for name, count in Counter(all).most_common():
        print(name, count)

    print('Unique values:', len(s))
    print(df['cerif_conf'].value_counts())


def preprocess_labels(cerif):
    mlb = MultiLabelBinarizer()
    labels = [set(c.split('|')) for c in cerif]
    y = mlb.fit_transform(labels)
    print(mlb.classes_)
    return y


def preprocess_data(table, save_to_file, nlp_tool, precalculated_lemmas):
    def build_lemmas(path):
        w2l = {}
        with open(path, 'r') as f:
            for line in tqdm(f):
                word, lemma, _, _ = line.strip().split('\t')
                w2l[word] = lemma
            return w2l

    def get_body(file, root='data/kas.txt'):
        full_path = os.path.join(root, file)
        with open(full_path) as f:
            text = f.read()
        return text

    def process_body_corpus_beseda(text, stopwords, punct, lemmas):
        tokens = word_tokenize(text, language='slovene')
        filtered_text = []
        for word in tokens:
            word = word.lower()
            if word in stopwords or word in punct:
                continue
            elif any(char.isdigit() for char in word):
                continue
            else:
                lemma = lemmas.get(word)
                if lemma:
                    filtered_text.append(lemma)
                else:
                    filtered_text.append(word)
        text = ' '.join(filtered_text)
        text = text.replace(u'\xa0', u' ')
        return text

    def process_body_classla(text, stopwords, punct):
        doc = nlp_tool(text)
        filtered_text = []
        for sent in doc.sentences:
            for word in sent.words:
                word_lemma = word.lemma.lower()
                if word_lemma in stopwords or word_lemma in punct:
                    continue
                elif any(char.isdigit() for char in word_lemma):
                    continue
                else:
                    filtered_text.append(word_lemma)
        text = ' '.join(filtered_text)
        text = text.replace(u'\xa0', u' ')
        return text

    # import main table
    df = pd.read_table(table)
    # basic_info(df)

    # select high subcerif levels, add bodies to df, preprocess bodies
    df_subcerif = df[df['cerif_conf'] == 'high']

    df_subcerif['body'] = df_subcerif['file'].progress_apply(get_body)
    print('Body added ...')

    # prepare helpers for preprocessing
    stopwords = {w.strip() for w in open('slovene-stopwords.txt', encoding='utf-8').read().splitlines()}
    punct = string.punctuation

    if precalculated_lemmas:
        lemmas = build_lemmas('resources/Beseda_Corpus_Lemmatisation_Lexicon.txt')
        df_subcerif['processed_body'] = df_subcerif['body'].progress_apply(process_body_corpus_beseda, stopwords=stopwords, punct=punct,
                                                                           lemmas=lemmas)
    else:
        df_subcerif['processed_body'] = df_subcerif['body'].progress_apply(process_body_classla, stopwords=stopwords, punct=punct)

    print('Text processed ...')

    # save processed file
    df_subcerif.to_csv(save_to_file, index=False)


def tfidf(df_subcerif):
    start_time = time.time()
    tfidf = TfidfVectorizer()
    tfidf.fit(df_subcerif['processed_body'])
    X = tfidf.transform(df_subcerif['processed_body'])
    print("--- %s seconds ---" % (time.time() - start_time))
    return X


def bit_accuracy(y_true, y_pred):
    number_of_examples = y_true.shape[0] * y_true.shape[1]
    error = np.sum(np.abs(y_true - y_pred))
    return (number_of_examples - error) / number_of_examples


def pattern_accuracy(y_true, y_pred):
    return np.sum(np.all(y_true == y_pred, axis=1)) / y_true.shape[0]


def basic_classifiers(X_train, X_test, y_train, y_test):
    classifiers = {
        'LR': LogisticRegression(),
        'KN': KNeighborsClassifier(),
        'SVM': LinearSVC(),
    }

#     parameters = {
#         'LR': {'estimator__solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]},
#         'KN': {'estimator__n_neighbors': [5, 15, 50],
#                'estimator__weights': ['uniform', 'distance']},
#         'SVM': {'estimator__kernel' : ["linear", "poly", "rbf", "sigmoid", "precomputed"]}
# }

    for name, clf in classifiers.items():
        # train
        print(name)
        clf = MultiOutputClassifier(clf, n_jobs=-1).fit(X_train, y_train)

        # gs = GridSearchCV(MultiOutputClassifier(clf), param_grid=parameters[name], verbose=3, n_jobs=-1)
        # gs.fit(X_train, y_train)
        # print(gs.best_estimator_, gs.best_params_, gs.best_score_)

        # tests
        y_pred = clf.predict(X_test)

        print(pattern_accuracy(y_test, y_pred))
        print(bit_accuracy(y_test, y_pred))
        print('\n' * 2)


def mlp(X_train, X_test, y_train, y_test):
    filename = 'output/cerif_models/baseline_subcerif.pkl'

    # # save the model to disk
    clf = MLPClassifier(random_state=1, max_iter=300, verbose=1).fit(X_train, y_train)
    pickle.dump(clf, open(filename, 'wb'))

    # load the model from disk
    clf = pickle.load(open(filename, 'rb'))

    # tests
    y_pred = clf.predict(X_test)

    print(pattern_accuracy(y_test, y_pred))
    print(bit_accuracy(y_test, y_pred))


def deep_mlp(X_train, X_test, y_train, y_test):
    X_train, X_test = X_train.todense(), X_test.todense()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(y_train.shape[1], kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=4,
                        epochs=300,
                        verbose=1,
                        validation_split=.2,
                        callbacks=[callback])

    model.save('output/cerif_models/baseline_subcerif_keras.h5')
    model = load_model('output/cerif_models/baseline_subcerif_keras.h5')

    y_class = model.predict_classes(X_test)
    y_prob = model.predict_proba(X_test)

    # reports
    score = model.evaluate(X_test, y_test, batch_size=32)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])


def deep_mlp_batch(X_train, X_test, y_train, y_test):
    def batch_generator(X, y, batch_size, shuffle):
        number_of_batches = np.ceil(X.shape[0] / batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :].toarray()
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

    def batch_generator_pred(X, batch_size, shuffle):
        number_of_batches = np.ceil(X.shape[0] / batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :].toarray()
            counter += 1
            yield X_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

    # validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(y_validation.shape[0])

    # model config
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='output/cerif_models/baseline_subcerif_keras_generator.{epoch:02d}-{val_loss:.5f}.h5',
        save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='output/cerif_models/logs/'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    ]

    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dense(y_train.shape[1], kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()


    model.fit_generator(generator=batch_generator(X_train, y_train, 32, False),
                        epochs=100,
                        steps_per_epoch=X_train.shape[0] // 32,
                        validation_data=(X_validation.todense(), y_validation),
                        validation_freq=1,
                        verbose=1,
                        callbacks=my_callbacks
                        )

    # # load model
    # model = load_model('/home/azagar/myfiles/kas/output/cerif_models/baseline_subcerif_keras_generator.14-0.03773.h5')

    # reports
    y_pred = (model.predict(X_test.toarray()) > 0.5).astype("int32")

    accuracy_score(y_pred, y_test)
    print(pattern_accuracy(y_test, y_pred))
    print(bit_accuracy(y_test, y_pred))


def save_tfidf(X, file):
    with open(file, 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tfidf(file):
    with open(file, 'rb') as handle:
        X = pickle.load(handle)
    return X


def save_labels(y, task, folder):
    with open(f'{folder}/y-{task}.pkl', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_labels(task, folder):
    with open(f'{folder}/y-{task}.pkl', 'rb') as handle:
        y = pickle.load(handle)
    return y


def main():
    # select task (subcerif or cerif)
    TASK = 'subcerif'

    # classla.download('sl')  # download standard models for Slovenian, use hr for Croatian, sr for Serbian, bg for Bulgarian, mk for Macedonian
    nlp = classla.Pipeline('sl', use_gpu=True, processors='tokenize,lemma')
    #### 1 preprocess data (filter stopwords, create lemmas, ...) ####
    preprocess_data('data/kas-meta-texts.tbl',
                    save_to_file='output/cerif_preprocessed/classla/df_subcerif_classla.csv',
                    nlp_tool=nlp,
                    precalculated_lemmas=False)

    # #### 2 load processed file ####
    csv = 'output/cerif_preprocessed/classla/df_subcerif_classla.csv'
    df = pd.read_csv(csv)
    print('Data imported ... ')

    #### 3 TFIDF ####
    # prepare or reload tfidf
    file = 'output/cerif_preprocessed/classla/X-default.pkl'
    X = tfidf(df)
    print('TF-IDF prepared ... ')
    save_tfidf(X, file=file)
    print('TF-IDF saved ... ')

    X = load_tfidf(file=file)
    print('Preprocessed TF-IDF loaded ... ')

    #### 4 Labels ####
    # prepare or reload labels
    folder = 'output/cerif_preprocessed/classla'
    y = preprocess_labels(df[TASK].tolist())
    print('Labels prepared ... ')
    save_labels(y, TASK, folder=folder)
    print(f'{TASK} labels saved ... ')

    y = load_labels(TASK, folder=folder)
    print(f'{TASK} labels loaded ... ')

    # # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # train (no big difference between mlp and deep mlp)
    basic_classifiers(X_train, X_test, y_train, y_test)
    # mlp(X_train, X_test, y_train, y_test)
    # deep_mlp(X_train, X_test, y_train, y_test)
    # deep_mlp_batch(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
