#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012 Herve BREDIN (bredin@limsi.fr)

# This file is part of PyAnnote.
#
#     PyAnnote is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     PyAnnote is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with PyAnnote.  If not, see <http://www.gnu.org/licenses/>.

import itertools

import numpy as np
import scipy.signal
import scipy
from scipy.misc import logsumexp
import sklearn
from pyannote import Timeline
from pyannote.base.segment import Segment, SlidingWindow
from pyannote.stats.gaussian import Gaussian
from sklearn.mixture import GMM
import networkx as nx


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


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

    def diff(self, left, right, feature):
        """Compute difference between left and right windows

        Parameters
        ----------
        left, right : Segment
            Left and right windows
        feature : SlidingWindowFeature
            Pre-extracted features

        Returns
        -------
        d : float
            Difference value (the higher, the more different)
        """
        raise NotImplementedError(
            'You must inherit from SlidingWindowSegmentation')

    def iterdiff(self, feature):
        """(middle, difference) generator

        `middle`
        `difference`


        Parameters
        ----------
        feature : SlidingWindowFeature
            Pre-extracted features
        """

        #
        focus = feature.getExtent()

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=focus.start, end=focus.end)

        for left in sliding_window:

            right = Segment(
                start=left.end,
                end=left.end + self.duration + self.gap
            )
            middle = .5*(left.end + right.start)

            yield middle, self.diff(left, right, feature)

    def iterdiff_STG(self, feature, subsegments, thresold_time):

        # Thresold_time : seuil sur temps
        mat_dissimilarity = np.zeros((len(subsegments), len(subsegments)))

        for l, left in enumerate(subsegments):
            for r, right in enumerate(subsegments):
                if (left ^ right).duration < thresold_time:
                    #print left, right
                    mat_dissimilarity[l, r] = self.diff(left, right, feature)
                else:
                    mat_dissimilarity[l, r] = np.inf
        return mat_dissimilarity
        # return middle, mat_dissimilarity

    def apply(self, feature):

        x, y = zip(*[
            (m, d) for m, d in self.iterdiff(feature)
        ])
        x = np.array(x)
        y = np.array(y)

        # find local maxima
        maxima = scipy.signal.argrelmax(y)
        x = x[maxima]
        y = y[maxima]

        # only keep high enough local maxima
        high_maxima = np.where(y > self.threshold)

        # create list of segment boundaries
        # do not forget very first and last boundaries
        extent = feature.getExtent()
        boundaries = itertools.chain(
            [extent.start], x[high_maxima], [extent.end]
        )

        # create list of segments from boundaries
        segments = [Segment(*p) for p in pairwise(boundaries)]

        # TODO: find a way to set 'uri'
        return Timeline(segments=segments, uri=None)

    def get_maxima(self, feature):
        x, y = zip(*[
            (m, d) for m, d in self.iterdiff(feature)
        ])
        x = np.array(x)
        y = np.array(y)
        # find local maxima
        maxima = scipy.signal.argrelmax(y)
        x = x[maxima]
        y = y[maxima]
        print x, y
        return x, y

    def apply_threshold(self, feature, x, y):

        # only keep high enough local maxima
        high_maxima = np.where(y > self.threshold)

        # create list of segment boundaries
        # do not forget very first and last boundaries
        extent = feature.getExtent()
        boundaries = itertools.chain(
            [extent.start], x[high_maxima], [extent.end]
        )

        # create list of segments from boundaries
        segments = [Segment(*p) for p in pairwise(boundaries)]

        print "plage de variation"
        print extent

        # TODO: find a way to set 'uri'
        return Timeline(segments=segments, uri=None)


class SegmentationGaussianDivergence(SlidingWindowsSegmentation):

    def __init__(
        self,
        duration=1., step=0.1, gap=0., threshold=0.
    ):

        super(SegmentationGaussianDivergence, self).__init__(
            duration=duration, step=step, gap=gap, threshold=threshold
        )

    def diff(self, left, right, feature):

        gl = Gaussian(covariance_type='diag')
        Xl = feature.crop(left)
        gl.fit(Xl)

        gr = Gaussian(covariance_type='diag')
        Xr = feature.crop(right)
        gr.fit(Xr)

        try:
            divergence = gl.divergence(gr)
        except:
            divergence = np.NaN

        return divergence
"""
==================================================================
            Méthode par Graphe de Transition
==================================================================
"""


class TransitionGraphSegmentation(object):

    def __init__(
        self, duration=15., similarity_th=5., temporal_th=120
    ):
        super(TransitionGraphSegmentation, self).__init__()
        self.duration = duration
        self.similarity_th = similarity_th
        self.temporal_th = temporal_th

    def diff(self, left, right, feature):

        gl = Gaussian(covariance_type='diag')
        Xl = feature.crop(left)
        gl.fit(Xl)

        gr = Gaussian(covariance_type='diag')
        Xr = feature.crop(right)
        gr.fit(Xr)
        try:
            divergence = gl.divergence(gr)
        except:
            if np.size(gl.covar) != 0.:
                divergence = 0.
                # det(gl.covar)
            else:
                divergence = np.NaN
        return divergence

    def sur_segmentation(self, feature):

        """ Segmentation initiale à l'entrée du STG"""

        focus = feature.getExtent()

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.duration,
            start=focus.start, end=focus.end)

        LEFT = []
        RIGHT = []
        for left in sliding_window:
            right = Segment(
                start=left.end,
                end=left.end + self.duration
            )
            LEFT.append(left)
            RIGHT.append(right)

        return LEFT, RIGHT

    def iterdiff(self, feature, first_segments):

        """Calcule de la matrice de similarité """

        mat_dissimilarity = np.zeros(
            (len(first_segments), len(first_segments))
        )

        for l, left in enumerate(first_segments):
            for r, right in enumerate(first_segments):
                if (left ^ right).duration < self.temporal_th:
                    #print left, right
                    mat_dissimilarity[l, r] = self.diff(left, right, feature)
                else:
                    mat_dissimilarity[l, r] = np.inf
        X = scipy.spatial.distance.squareform(mat_dissimilarity)
        Y = scipy.cluster.hierarchy.linkage(X, method='average')
        T = scipy.cluster.hierarchy.fcluster(
            Y, self.similarity_th, criterion='distance'
        )
        print T
        return T

    def build_nodes_edges(self, feature, first_segments):

        """Construction du graphe G"""

        T = self.iterdiff(feature, first_segments)
        i = 0
        G = nx.DiGraph()
        n_nodes = []

        while i < (len(T) - 1):
            if T[i] - T[i + 1] != 0:
                G.add_edge(T[i], T[i + 1])
                n_nodes.append(i)
            i += 1
        print n_nodes
        return G, n_nodes, T

    def cut_edges_detection(self, feature, first_segments):

        """Recherche des arcs de coupure
            first_segments: segmenation initiale"""

        G, n_nodes, T = self.build_nodes_edges(feature, first_segments)

        hyp = Timeline()
        hypothesis = Timeline()
        hyp.add(
            Segment(first_segments[0].start, first_segments[n_nodes[0]].end)
        )
        for i, j in enumerate(n_nodes):
            hyp.add(
                Segment(
                    first_segments[n_nodes[i - 1]].end,
                    first_segments[n_nodes[i]].end
                )
            )
            Coupure = nx.minimum_edge_cut(G, T[j + 1], T[j])
            if len(Coupure) == 0:
                hypothesis.add(hyp[i])

        return hypothesis

"""
==================================================================
            Méthode par Plus Court Chemin
==================================================================
"""


class PlusCourtChemin(object):

    """ Rechecher de la segmentation la plus probable"""

    def __init__(
        self, duration=5., step=5, gap=0., threshold=240.,
            n_components=1, covariance_type='diag', penality_coef=1.
    ):
        super(PlusCourtChemin, self).__init__()
        self.duration = duration
        self.step = duration
        self.gap = gap
        self.threshold = threshold
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.penality_coef = penality_coef

    """step = duration pour empêcher les recouvrements et d'avoir une cohesion
    sur un et un seul segment à chaque fois
    """

    def apply(self, feature):

        x, y = zip(*[
            (m, d) for m, d in self.iterdiff(feature)
        ])
        x = np.array(x)
        y = np.array(y)

        # find local maxima
        maxima = scipy.signal.argrelmax(y)
        x = x[maxima]
        y = y[maxima]

        # only keep high enough local maxima
        high_maxima = np.where(y > self.threshold)

        # create list of segment boundaries
        # do not forget very first and last boundaries
        extent = feature.getExtent()
        boundaries = itertools.chain(
            [extent.start], x[high_maxima], [extent.end]
        )

        # create list of segments from boundaries
        segments = [Segment(*p) for p in pairwise(boundaries)]

        # TODO: find a way to set 'uri'
        return Timeline(segments=segments, uri=None)

    def iterdiff(self, feature):
        """(middle, difference) generator

        `middle`
        `difference`


        Parameters
        ----------
        feature : SlidingWindowFeature
            Pre-extracted features
        """

        #
        focus = feature.getExtent()

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=focus.start, end=focus.end)

        for left in sliding_window:

            right = Segment(
                start=left.end,
                end=left.end + self.duration + self.gap
            )
            middle = .5*(left.end + right.start)

            yield middle, self.diff(left, right, feature)

    def sur_segmentation(self, feature):

        """ Segmentation initiale à l'entrée du STG"""

        focus = feature.getExtent()

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=focus.start, end=focus.end)

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

    def apply_thematique(self, feature):

        seg, _ = self.sur_segmentation(feature)
        graph = nx.Graph()
        graph.add_node(0, demand=1)
        graph.add_node(len(seg), demand=-1)

        for i, l in enumerate(seg):
            for j, r in enumerate(seg):
                if j >= i and (l ^ r).duration < self.threshold:

                    cohesion = self.bic_criterium(feature, (l | r))
                    graph.add_edge(i, j + 1, weight=abs(cohesion))
                    "attenetion (i,j+1)= 1er segment ! "

        Chemin = nx.shortest_path(graph, 0, len(seg), weight='weight')
        hypothesis = Timeline()
        hypothesis.add(seg[-1])
        for i in Chemin[:-1]:
            hypothesis.add(seg[i])

        return Chemin, hypothesis

    def log_multivariate_normal_density(self, X, means, covars):
        """Compute Gaussian log-density at X for a diagonal model"""
        n_samples, n_dim = X.shape
        lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1) - 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))

        return lpr

    def score_sample_adaptative(self, X, gaussian_):

        """ Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
        Log probabilities of each data point in X.


        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, gaussian_.n_components))
        if X.shape[1] != gaussian_.means_.shape[1]:
            raise ValueError('The shape of X is not compatible with self')

        lpr = self.log_multivariate_normal_density(
            X, gaussian_.means_, gaussian_.covars_
        )

        logprob = logsumexp((lpr + np.log(gaussian_.weights_)), axis=1)

        return logprob

    # =====================================

    def train_world_model(self, feature):

        self._world_model = GMM(
            n_components=self.n_components, covariance_type=self.covariance_type
        )
        self._world_model.fit(feature.data)

    def segment_model(self, segment, feature):

        """ return la gaussian adapetée au nouveau segment
        (adaptation des poids) """

        X = feature.crop(segment)
        # g = self.gw.copy()

        g = sklearn.clone(self._world_model)
        # g.params = 'w'
        g.n_iter = self._world_model .n_iter
        g.n_init = 1
        g.init_params = ''

        g.weights_ = self._world_model .weights_
        g.means_ = self._world_model .means_
        g.covars_ = self._world_model .covars_
        g.params = 'w'
        g.fit(X)

        return g, X

    def build_matrics(self, feature, segmentation):
        """Calcul de la matrice de cohésion et les coefficients
            de pénélyté"""

        self.train_world_model(feature)
        _, responsibilities = self._world_model.score_samples(feature.data)
        penality = self._world_model._n_parameters()
        print "calcul des matrices"
        """ matrice d'analyse des couts """
        mat_likelyhood = np.zeros((len(segmentation), len(segmentation)))
        mat_penality = np.zeros((len(segmentation), len(segmentation)))
        for i, l in enumerate(segmentation):
            for j, r in enumerate(segmentation):
                if j >= i:
                    if (l ^ r).duration < self.threshold:
                        segment = l | r
                        print segment
                        g, x = self.segment_model(segment, feature)
                        logprob = self.score_sample_adaptative(x, g)

                        k, n = feature.sliding_window.segmentToRange(segment)

                        ''' la matrice de visualisation des couts '''
                        mat_likelyhood[i, j] = -(logprob.sum())
                        print "j'ai calculé la matrice de vraisemblance"
                        mat_penality[i, j] = penality * np.log(n)
                        print "j'ai fini les calculs avec succès"

        return mat_likelyhood, mat_penality

    def build_graph2(self, mat_likelyhood, mat_penality):

        """ matrice d'analyse des couts """
        mat_likelyhood[np.where(mat_likelyhood == 0)] = np.inf
        mat_penality[np.where(mat_penality == 0)] = np.inf

        matrice_cout = np.zeros((len(mat_likelyhood), len(mat_likelyhood)))
        matrice_cout = (mat_likelyhood + (self.penality_coef * mat_penality))

        matrice_cout = matrice_cout + 2 * abs(np.amin(matrice_cout))

        Graph = nx.DiGraph()
        Graph.add_node(0)
        Graph.add_node(len(matrice_cout))

        for i in range(len(matrice_cout)):
            for j in range(len(matrice_cout)):
                Graph.add_edge(i, j + 1, weight=matrice_cout[i, j])

        Chemin = nx.shortest_path(Graph, 0, len(matrice_cout), weight='weight')
        nbre_segment = len(Chemin)
        print nbre_segment

        return Chemin

    def apply_gtg(self, mat_likelyhood, mat_penality, segmentation):
        Chemin = self.build_graph2(mat_likelyhood, mat_penality)
        print "je calcul l'hypothesis"
        hypothesis = Timeline()
        hypothesis.add(segmentation[-1])
        for i, l in enumerate(Chemin[:-2]):
            hypothesis.add(Segment(
                segmentation[l].start, segmentation[Chemin[i + 1]].start)
            )
        hypothesis.add(
            Segment(segmentation[Chemin[i + 1]].start, segmentation[-1].start)
        )

        return Chemin, hypothesis
