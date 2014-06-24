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

from pyannote import Timeline
from pyannote.base.segment import Segment, SlidingWindow
from pyannote.stats.gaussian import Gaussian
from sklearn.mixture import GMM


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
        return x, y

    def get_maximaSTG(self, feature):
        focus = feature.getExtent()
        x = []
        y = []
        x = self.iterdiff_STG(feature)
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


class SegmentationThematique(SlidingWindowsSegmentation):
    """docstring for SegmentationMamadou"""
    def __init__(
        self, duration=1., step=0.1, gap=0., threshold=0.
    ):
        super(SegmentationThematique, self).__init__(
            duration=duration, step=step, gap=gap, threshold=threshold
        )

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


    def sur_segmentation(self, feature,duration):

        # la longueur total du flux audio
        focus = feature.getExtent()
        #start=focus.start
        #print focus

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=focus.start, end=focus.end)
        #print sliding_window


        LEFT = []
        RIGHT = []
        for left in sliding_window:
            right = Segment(
                start=left.end,
                end=left.end + self.duration + self.gap
            )
            # Pb:  left et right n'incremente pas !!!
            # middle = .5*(left.end + right.start)
            # LEFT = feature.crop(left)
            # RIGHT = feature.crop(right)
            # chaque intervalle contient 936 verctuers
            # et chaque vecteur contient le 12 coef MCFF 
            #    + 1 coef d'Egi
            # print left, right

            LEFT.append(left)
            RIGHT.append(right)
            #print left, right

            yield LEFT, RIGHT

            #yield middle, self.bic_cout(left, right, feature)
           
    
    def apply_thematique(self, feature, duration, gamma, n_Gauss, threshold):

        #x, y = zip(*[
        #    (m, b) for m, b in self.sur_segmentation(feature)
        #])
        #x = np.array(x)
        #y = np.array(y)

        
        
        for sgmtl, sgmtr in self.sur_segmentation(feature,duration):
            left = sgmtl
        seg = left
       
        mat=np.zeros((len(seg),len(seg)))
        for i,l in enumerate(seg):
            for j,r in enumerate(seg):
                if j>=i and (l ^ r).duration < threshold:
                    #c = l | r
                    mat[i,j] = self.bic_criterium(feature, (l | r),gamma, n_Gauss)

        
        return mat, seg

    def bic_criterium(self, feature, segment,gamma, n_components):

        # fait en sorte qu'il prend
        # à l'entrée nbre de gaussiens
        g = GMM(n_components=n_components)
        #X = feature.crop(segment)
        # g.fit(X)

        # Reduction du nbre de point pour aug le tps de calcul
        X = feature.crop(segment)
        # en prenant 1/10ieme des points
        X_reduit = X[::10, :]
        g.fit(X_reduit)
        # en prenant la mat de covariance
        #g.fit(X)
        #X_cov = np.round(g.means_, 4)
        

        #bicc = g.bic(X) + gamma * (g._n_parameters() * np.log(X.shape[0]))

        bicc = (-2 * g.score(X_reduit).sum() +
                g._n_parameters() * gamma * np.log(X.shape[0]))

        return bicc

