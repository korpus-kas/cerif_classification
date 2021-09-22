import os
import pickle
import string
import time

import classla
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tqdm.pandas()


# preprocess data, load models
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
    df_subcerif = df[df['cerif_conf'] == 'low']

    df_subcerif['body'] = df_subcerif['file'].progress_apply(get_body)
    print('Body added ...')

    # prepare helpers for preprocessing
    stopwords = {w.strip() for w in open('slovene-stopwords.txt', encoding='utf-8').read().splitlines()}
    punct = string.punctuation

    if precalculated_lemmas:
        lemmas = build_lemmas('resources/Beseda_Corpus_Lemmatisation_Lexicon.txt')
        df_subcerif['processed_body'] = df_subcerif['body'].progress_apply(process_body_corpus_beseda,
                                                                           stopwords=stopwords, punct=punct,
                                                                           lemmas=lemmas)
    else:
        df_subcerif['processed_body'] = df_subcerif['body'].progress_apply(process_body_classla, stopwords=stopwords,
                                                                           punct=punct)

    print('Text processed ...')

    # save processed file
    df_subcerif.to_csv(save_to_file, index=False)


# vectorize, calculate tf-idf values
def tfidf(df_subcerif):
    start_time = time.time()

    # load model from disk
    with open('output/cerif_final_models/tfidf-processor.obj', 'rb') as fp:
        tfidf = pickle.load(fp)

    # transform
    X = tfidf.transform(df_subcerif['processed_body'])
    print("--- %s seconds ---" % (time.time() - start_time))
    return X


def predict_labels(X):
    # load model
    with open(f'output/cerif_final_models/SVM.obj', 'rb') as fp:
        clf = pickle.load(fp)
        y_pred = clf.predict(X)

    # load labels processor
    with open(f'output/cerif_final_models/labels-processor.obj', 'rb') as fp:
        mlb = pickle.load(fp)
        labels = mlb.inverse_transform(y_pred)
        formated_labels = ["|".join(l) if l else np.nan for l in labels]

    return formated_labels


def main():
    #### 0 import NLP tools ####
    classla.download('sl')
    nlp = classla.Pipeline('sl', use_gpu=True, processors='tokenize,lemma')

    #### 1 preprocess learning data (filter stopwords, create lemmas, ...) ####
    preprocess_data('data/kas-meta-texts.tbl',
                    save_to_file='output/cerif_final_models/df_preprocessed_cerif2predict_Beseda_Corpus.csv',
                    nlp_tool=nlp,
                    precalculated_lemmas=False)

    #### 2 load processed file ####
    csv = 'output/cerif_final_models/df_preprocessed_cerif2predict_Beseda_Corpus.csv'
    df = pd.read_csv(csv)
    print('Data imported ... ')

    #### 3 TFIDF ####
    X = tfidf(df)

    #### 4 Predict labels ####
    predicted_labels = predict_labels(X)

    #### 5 Export csv file with predicted subcerif ####
    df['predicted_subcerif'] = predicted_labels
    df[['file', 'predicted_subcerif']].to_csv('file-with-predicted-subcerif.csv', index=False)


if __name__ == '__main__':
    main()
