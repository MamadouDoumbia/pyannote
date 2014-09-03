#!/usr/bin/env python
# encoding: utf-8

import itertools
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyannote import Timeline
from pyannote.base.segment import Segment, SlidingWindow
import networkx as nx
import nltk
from nltk.collocations import *
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import ADJ, NOUN, ADV, VERB


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

"""
==================================================================
            Méthode par Plus Court Chemin
==================================================================
"""


class GraphGlobal(object):
    """docstring for SegmentationMamadou"""
    def __init__(
        self, duration=5., step=5, gap=0., threshold=240.,
            penality_coef=1.
    ):
        super(GraphGlobal, self).__init__()
        self.duration = duration
        self.step = duration
        self.gap = gap
        self.threshold = threshold
        self.penality_coef = penality_coef

    """stpe = duration pour empêcher les recouvrements et d'avoir une cohesion
    sur un et un seul segment à chaque fois
    """

    def sur_segmentation(self, feature):

        # focus = feature.getExtent()
        focus = feature[len(feature) - 1].end

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.duration,
            start=0, end=focus)

        LEFT = []
        RIGHT = []
        for left in sliding_window:
            right = Segment(
                start=left.end,
                end=left.end + self.duration + self.gap
            )
            LEFT.append(left)
            RIGHT.append(right)

        return LEFT, RIGHT

    def text_segmentation(self, feature):

        # la longueur total du text
        feature_start = float(feature[2][2])
        feature_end = float(feature[-1][2])

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=feature_start, end=feature_end)
        print sliding_window

        LEFT = []
        RIGHT = []
        for left in sliding_window:
            right = Segment(
                start=left.end,
                end=left.end + self.duration + self.gap
            )

            LEFT.append(left)
            RIGHT.append(right)

        return LEFT, RIGHT

    def text_weights(self, text, total_words_epi, vocab_epi):

        lmtz = nltk.stem.wordnet.WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        lemmas = {}
        total_words = 0.
        keep = {'JJ': ADJ,'JJR': ADJ,'JJS': ADJ,'NN': NOUN,'NNP': NOUN,'NNPS': NOUN,'NNS': NOUN,'RB': ADV,'RBR': ADV,'RBS': ADV,'VB': VERB,'VBD': VERB,'VBG': VERB,'VBN': VERB,'VBP': VERB,'VBZ': VERB}
        for w, pos in tags:
            if pos == 'NNS' or pos == 'NNP' or pos == 'NN' or pos == 'VBD' or pos == 'VBZ' or pos == 'VB' or pos == 'VBG' or pos == 'VBZ' or pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
                l = lmtz.lemmatize(w.lower(), pos=keep[pos])
                total_words += 1
                if l.lower() in lemmas:
                    lemmas[l.lower()] = lemmas[l.lower()] + 1
                else:
                    lemmas[l.lower()] = 1
        """calcul des couts)"""
        likelihood = []
        K = total_words_epi
        #for word in vocab_epi:
            #if word in lemmas:
                #word_weights = ((lemmas[word] + 1) / (total_words + K))
            #else:
                #word_weights = (1 / (total_words + K))
            #likelihood.append(np.log(word_weights))
        for word in lemmas:
            word_weights = ((lemmas[word] + 1) / (total_words + K))
            likelihood.append(np.log(word_weights))
        segment_likelihood = sum(likelihood)

        return segment_likelihood

    def episode_penality2(self, lines_sentence):
        lmtz = nltk.stem.wordnet.WordNetLemmatizer()
        text = ""
        for _, line in enumerate(lines_sentence[2:]):
            text = str(text + " " + line[4])

        ''' nbres_words = nbre total des differents mots dans le texte  '''
        total_words_epi = 0.

        ''' nbres_words = nbre total de mots avec les repetitions'''
        nbres_words = 0.
        vocab_epi = {}
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        keep = {'JJ': ADJ,'JJR': ADJ,'JJS': ADJ,'NN': NOUN,'NNP': NOUN,'NNPS': NOUN,'NNS': NOUN,'RB': ADV,'RBR': ADV,'RBS': ADV,'VB': VERB,'VBD': VERB,'VBG': VERB,'VBN': VERB,'VBP': VERB,'VBZ': VERB}

        ''' si le mots est l'un des pos alors on ajoute
        au dictionnaire '''
        for w, pos in tags:
            if pos == 'NNS' or pos == 'NNP' or pos == 'NN' or pos == 'VBD' or pos == 'VBZ' or pos == 'VB' or pos == 'VBG' or pos == 'VBZ' or pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
                nbres_words += 1
                l = lmtz.lemmatize(w.lower(), pos=keep[pos])
                if l.lower() in vocab_epi:
                    vocab_epi[l.lower()] = vocab_epi[l.lower()] + 1
                else:
                    vocab_epi[l.lower()] = 1

        total_words_epi = len(vocab_epi)
        # print total_words_epi
        penality = np.log(nbres_words)

        return penality, total_words_epi, vocab_epi

    def construction_graph(self, lines_sentence, segmentation):

        text = ""
        """ terme penalisant"""
        #P = self.episode_penality(lines_sentence)
        penality, total_words_epi, vocab_epi = self.episode_penality2(lines_sentence)
        graph = nx.DiGraph()
        graph.add_node(0, demand=1)
        graph.add_node(len(segmentation), demand=-1)
        """ matrice d'analyse des couts """
        mat = np.zeros((len(segmentation), len(segmentation)))
        for i, l in enumerate(segmentation):
            for j, r in enumerate(segmentation):
                if j >= i:
                    if (l ^ r).duration < self.threshold:
                        #segmentation = l | r
                        for _, line in enumerate(lines_sentence):
                            # prendre juste les mots ds (l | r)
                            if float(line[2]) >= (l | r).start and float(line[2]) <= (l | r).end:
                                # print (l | r), line[4]
                            # if float(line[2]) < (l | r).end:
                                text = str(text + " " + line[4])
                                """ il manque le coef de penality"""
                        #print "je calcule le cout"
                        cout = -(self.text_weights(
                            text, total_words_epi, vocab_epi) - (
                            self.penality_coef * penality)
                        )

                        #print cout, self.text_weights(text, total_words_epi, vocab_epi), penality
                        #, self.text_weights(text), P
                        mat[i, j] = cout
                        graph.add_edge(i, j + 1, weight=cout)
                        ###print cout
                    else:
                        graph.add_edge(i, j + 1, weight=np.inf)
                        mat[i, j] = np.inf
                    # visualisation des differentes mots de (l | r)
                    # print text
                    # print " "
                    text = ""

        Chemin = nx.shortest_path(graph, 0, len(segmentation), weight='weight')
        print "chemin calculé:", Chemin
        hypothesis = Timeline()
        '''Il faut ajourer le 1er segment vu que la parole commence
        souvent après le générique et ce segment n'est pas ajouter
        au fichier ctm'''
        hypothesis.add(Segment(0, segmentation[0].start))
        hypothesis.add(segmentation[-1])
        for i, l in enumerate(Chemin[:-2]):
            hypothesis.add(Segment(
                segmentation[l].start, segmentation[Chemin[i + 1]].start)
            )
        hypothesis.add(
            Segment(segmentation[Chemin[i + 1]].start, segmentation[-1].start)
        )

        return Chemin, hypothesis, mat

"""
==================================================================
            Méthode par Fenêtre Glissante
==================================================================
"""


class SlidingWindowsSegmentation(object):
    """

    <---d---><-g-><---d--->
    [   L   ]     [   R   ]
         [   L   ]     [   R   ]
    <-s->

    Parameters
    ----------
    duration : float, optional
        Set left/right window duration. Defaults to 1 second.
    step : float, optional
        Set step duration. Defaults to 100ms
    gap : float, optional
        Set gap duration. Defaults to no gap (i.e. 0 second)
    """

    def __init__(self, duration=1.0, step=0.1, gap=0.0, threshold=0.):
        super(SlidingWindowsSegmentation, self).__init__()
        self.duration = duration
        self.step = step
        self.gap = gap
        self.threshold = threshold

    def features_extract(self, lines):

        '''f = codecs.open('/PATH2URI/uri.ctm', 'r', 'utf8')
            lines = [line.strip().split() for line in f.readlines()]
        '''
        start = float(lines[2][2])
        end = float(lines[-1][2])

        transcripts = {}
        corpus = []
        for i in range(2, len(lines)):
            transcripts[float(lines[i][2])] = lines[i][4]
        times = transcripts.keys()

        segments = Timeline()
        for i in range(int(start), int(end), self.step):
            start_segment = i
            end_segment = i + self.duration
            text = ""
            segments.add(Segment(start_segment, end_segment))

            for j, time in enumerate(times):
                words = transcripts[time]

                if time >= start_segment and time <= end_segment:
                    text = text + " " + words

            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            lemmas = self.clean(tags)
            corpus.append(lemmas)
        print "j'ai extrais les mots et le segments"

        return segments, corpus

    def clean(self, tags):
        lemmas = []
        lmtz = nltk.stem.wordnet.WordNetLemmatizer()
        keep = {'JJ': ADJ,'JJR': ADJ,'JJS': ADJ,'NN': NOUN,'NNP': NOUN,'NNPS': NOUN,'NNS': NOUN,'RB': ADV,'RBR': ADV,'RBS': ADV,'VB': VERB,'VBD': VERB,'VBG': VERB,'VBN': VERB,'VBP': VERB,'VBZ': VERB}

        for w, pos in tags:
            if pos == 'NNS' or pos == 'NNP' or pos == 'NN' or pos == 'VBD' or pos == 'VBZ' or pos == 'VB' or pos == 'VBG' or pos == 'VBZ' or pos=='JJ' or pos=='JJR' or pos == 'JJS':
                l = lmtz.lemmatize(w.lower(), pos=keep[pos])
                lemmas.append(l.lower())

        return ' '.join(lemmas)

    def iterdiff(self, segments, corpus):
        """(middle, difference) generator

        `middle`
        `difference`
        --------------------------------------------------

           <---d---><-g-><---d--->
            [   L1   ]     [   R1   ]
                [   L2   ]     [   R2   ]
            <-s->


        Le corpus est organisé de manière que les mots
        du segment R1 qui debute directement à la fin de L1
        est decalé à la (self.duration + 1)ieme iteration
        et c'est L1 et R1 qu'il faut comparer pour la MFG.
        Donc TFIDF[i] et TFIDF[self.duration + i +1]

        car TFIDF[i] et TFIDF[i + 1] correspond à L1 et L2 !

        """

        #
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(corpus)
        similarity = []
        TFIDF = tfidf.toarray()
        middle = []

        for i in range(len(segments) - (int(self.duration) + 1)):
            sim = self.cos_diff(TFIDF[i], TFIDF[self.duration + i + 1])
            similarity.append(sim)
            middle.append(
                np.mean(
                    [segments[i].start, + segments[self.duration + i + 1].end]
                )
            )

        return middle, similarity

    def apply_similarity(self, segments, corpus):

        x, y = self.iterdiff(segments, corpus)
        x = np.array(x)
        y = np.array(y)

        # find local maxima
        maxima = scipy.signal.argrelmax(y)
        x = x[maxima]
        y = y[maxima]

        high_maxima = np.where(y > self.threshold)

        boundaries = itertools.chain(
            [0], x[high_maxima], [segments[len(corpus) - 1].end]
        )
        # create list of segments from boundaries
        segments = [Segment(*p) for p in pairwise(boundaries)]

        print len(segments)
        print "j'ai fini le calcul"
        return Timeline(segments=segments, uri=None)


class SegmentationCosinusSimilarity(SlidingWindowsSegmentation):

    def __init__(
        self,
        duration=1., step=0.1, gap=0., threshold=0.
    ):

        super(SegmentationCosinusSimilarity, self).__init__(
            duration=duration, step=step, gap=gap, threshold=threshold
        )

    def cos_diff(self, left_TFIDF, right_TFIDF):

        cos = cosine_similarity(left_TFIDF, right_TFIDF)[0][0]

        if (sum(left_TFIDF) == 0 or sum(right_TFIDF) == 0):
            cos = 1
        similarity = 1 - cos

        return similarity

        """
        ==================================================================
                    Méthode par Graphe de Transition
        ==================================================================
        """


class StgSegmentation(object):

    def __init__(
        self, duration=20., similarity_th=5., temporal_th=120
    ):
        super(StgSegmentation, self).__init__()
        self.duration = duration
        self.similarity_th = similarity_th
        self.temporal_th = temporal_th

    def clean(self, tags):
        lemmas = []
        lmtz = nltk.stem.wordnet.WordNetLemmatizer()
        keep = {'JJ': ADJ,'JJR': ADJ,'JJS': ADJ,'NN': NOUN,'NNP': NOUN,'NNPS': NOUN,'NNS': NOUN,'RB': ADV,'RBR': ADV,'RBS': ADV,'VB': VERB,'VBD': VERB,'VBG': VERB,'VBN': VERB,'VBP': VERB,'VBZ': VERB}

        for w, pos in tags:
            if pos == 'NNS' or pos == 'NNP' or pos == 'NN' or pos == 'VBD' or pos == 'VBZ' or pos == 'VB' or pos == 'VBG' or pos == 'VBZ' or pos=='JJ' or pos=='JJR' or pos == 'JJS':
                l = lmtz.lemmatize(w.lower(), pos=keep[pos])
                lemmas.append(l.lower())

        return ' '.join(lemmas)

    def cos_diff(self, left_TFIDF, right_TFIDF):

        cos = cosine_similarity(left_TFIDF, right_TFIDF)[0][0]

        if (sum(left_TFIDF) == 0 or sum(right_TFIDF) == 0):
            cos = 1
        similarity = 1 - cos

        return similarity

    def get_transciption(self, segments, features):

        """ features: fichier ctm avec les 2ière lignes"""

        transcripts = {}
        for i in range(2, len(features)):
            transcripts[float(features[i][2])] = features[i][4]

        times = transcripts.keys()

        return times, transcripts

    def iterdiff(self, segments, features):

        """
            segments: la première segmentation à l'entrée du StgSegmentation
            corpus: le fichier ctm
            l: segment (left)
            r: (right)


            =======================================================
            pour chaque segment "l" (plan en vidéo) de la segmentation à l'entrée
            on cherche les segments "r" les similaire à celui-ci.
            Donc je fait un fit du texte sur l ainsi que le segment de comparaison

            Une fois les mots obtenus après pré-traitement je calcule la difference
            cosinus puis cette valeur est stockée dans une matrice.

            à l'aide de cette matrice je fais le regroupement hierarchique puis
            le graphe est construit à la fin

        """
        text_l = ""
        text_r = ""
        corpus = []
        """ matrice """
        mat = np.zeros((len(segments), len(segments)))

        times, transcripts = self.get_transciption(segments, features)
        vectorizer = TfidfVectorizer(min_df=1)
        for i, l in enumerate(segments):
            for j, r in enumerate(segments):
                    if (l | r).duration <= self.temporal_th and i != j:
                        for k in range(0, len(times)):
                            time = times[k]
                            words = transcripts[time]
                            if (time >= l.start and time <= l.end):
                                text_l = text_l + " " + words
                            if (time >= r.start and time <= r.end):
                                text_r = text_r + " " + words

                        tokens_l = nltk.word_tokenize(text_l)
                        tags_l = nltk.pos_tag(tokens_l)
                        lemmas_l = self.clean(tags_l)
                        corpus.append(lemmas_l)
                        tokens_r = nltk.word_tokenize(text_r)
                        tags_r = nltk.pos_tag(tokens_r)
                        lemmas_r = self.clean(tags_r)
                        corpus.append(lemmas_r)

                        #print l," ",r
                        #print corpus
                        #print lemmas
                        text_l = ""
                        text_r = ""

                        if corpus != ['', ''] and lemmas_l != "!" and lemmas_r != "!" and lemmas_l != "'s" and lemmas_r != "'s":
                            #print i,j,
                            tfidf = vectorizer.fit_transform(corpus)
                            TFIDF = tfidf.toarray()

                            # mat[i, j] = self.cos_diff(TFIDF[0], TFIDF[1])
                            mat[i, j] = 1.1 - cosine_similarity(TFIDF[0], TFIDF[1])[0][0]
                        corpus = []
                        TFIDF = []
                    elif corpus == ["'s", ''] and lemmas_l == "'s" and lemmas_r == "'s":
                        mat[i, j] = 1.
                        " si pas de texte fichier vide on deside"
                        "qu'ils sont similaire"
                    elif i == j:
                        mat[i, j] = 0.
                    else:
                        mat[i, j] = np.inf
        "l'idéal serait de stocker la matrice similairité"
        X = scipy.spatial.distance.squareform(mat)
        Y = scipy.cluster.hierarchy.linkage(X, method='average')
        T = scipy.cluster.hierarchy.fcluster(
            Y, self.similarity_th, criterion='distance'
        )
        print "calcul de la matrice des couts terminé"
        print T
        return T

    def build_nodes_edges(self, segments, feature):

        print " construction des noeuds"
        T = self.iterdiff(segments, feature)
        i = 0
        G = nx.DiGraph()
        n_nodes = []

        while i < (len(T) - 1):
            if T[i] - T[i + 1] != 0:
                G.add_edge(T[i], T[i + 1])
                n_nodes.append(i)
                print T[i], T[i + 1],
            i += 1
        print n_nodes
        return G, n_nodes, T

    def cut_edges_detection(self, segments, feature):

        #T = self.iterdiff(feature, segments)
        G, n_nodes, T = self.build_nodes_edges(segments, feature)

        hyp = Timeline()
        hypothesis = Timeline()
        hyp.add(
            Segment(segments[0].start, segments[n_nodes[0]].end)
        )

        hyp.add(
            Segment(segments[0].start, segments[n_nodes[0]].end)
        )
        for i, j in enumerate(n_nodes):
            hyp.add(
                Segment(
                    segments[n_nodes[i - 1]].end,
                    segments[n_nodes[i]].end
                )
            )
            Coupure = nx.minimum_edge_cut(G, T[j + 1], T[j])
            if len(Coupure) == 0:
                hypothesis.add(hyp[i])

        return hypothesis
