#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012-2014 CNRS (Herve BREDIN -- bredin@limsi.fr)

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


from base import BaseMetric
from pyannote.base.annotation import Annotation
import numpy as np

PTY_NAME = 'segmentation purity'
CVG_NAME = 'segmentation coverage'
TOTAL = 'total'
INTER = 'intersection'
# # """ Mamadou"""
PRECISION_NAME = 'segmentation precision'
RECALL_NAME = 'segmentation recall'


class SegmentationCoverage(BaseMetric):
    """Segmentation coverage

    >>> from pyannote import Timeline, Segment
    >>> from pyannote.metric.segmentation import SegmentationCoverage
    >>> cvg = SegmentationCoverage()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> cvg(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> cvg(reference, hypothesis)
    0.75
    """

    @classmethod
    def metric_name(cls):
        return CVG_NAME

    @classmethod
    def metric_components(cls):
        return [TOTAL, INTER]

    def _get_details(self, reference, hypothesis, **kwargs):

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self._init_details()

        prev_r = None
        duration = 0.
        intersection = 0.
        for r, h in reference.co_iter(hypothesis):

            if r != prev_r:
                detail[TOTAL] += duration
                detail[INTER] += intersection
                duration = r.duration
                intersection = 0.
                prev_r = r
            intersection = max(intersection, (r & h).duration)

        detail[TOTAL] += duration
        detail[INTER] += intersection
        return detail

    def _get_rate(self, detail):

        return detail[INTER] / detail[TOTAL]

    def _pretty(self, detail):
        string = ""
        string += "  - duration: %.2f seconds\n" % (detail[TOTAL])
        string += "  - correct: %.2f seconds\n" % (detail[INTER])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


class SegmentationPurity(SegmentationCoverage):
    """Segmentation purity
    
    >>> from pyannote import Timeline, Segment
    >>> from pyannote.metric.segmentation import SegmentationPurity
    >>> pty = SegmentationPurity()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 1))
    >>> hypothesis.add(Segment(1, 2))
    >>> hypothesis.add(Segment(2, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> pty(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> pty(reference, hypothesis)
    0.5

    """
    @classmethod
    def metric_name(cls):
        return PTY_NAME

    def _get_details(self, reference, hypothesis, **kwargs):
        return super(SegmentationPurity, self)._get_details(
            hypothesis, reference, **kwargs
        )


class SegmentationPrecision(BaseMetric):
    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls):
        return ['Matches', 'Total']

    def __init__(self, tolerance=10., **kwargs):

        super(SegmentationPrecision, self).__init__()
        self.tolerance = tolerance

    def _get_details(self, reference, hypothesis, **kwargs):
        if isinstance(reference, Annotation):
            reference = reference.get_timeline()
        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()
        detail = self._init_details()
        nMatches = 0.

        N = len(reference) - 1
        M = len(hypothesis) - 1
        if N == 0.:
            if M == 0.:
                return 1.
            else:
                return 0.
        else:

            refBoundaries = [segment.end for segment in reference][:-1]
            hypBoundaries = [segment.end for segment in hypothesis][:-1]
            delta = np.zeros((N, M))
            for r, refBoundary in enumerate(refBoundaries):
                for h, hypBoundary in enumerate(hypBoundaries):
                    delta[r, h] = abs(refBoundary - hypBoundary)
            delta_max = np.amax(delta)
            delta[np.where(delta > self.tolerance)] = np.inf
            # print delta
            h = np.amin(delta)

            while h < delta_max:
                k = np.argmin(delta)
                i = k / M
                j = k % M
                delta[i, :] = np.inf
                delta[:, j] = np.inf
                nMatches += 1
                h = np.amin(delta)
            detail['Matches'] = nMatches
            detail['Total'] = len(hypBoundaries)
            return detail

    def _get_rate(self, detail):
        return detail['Matches'] / detail['Total']


class SegmentationRecall(SegmentationPrecision):

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    def _get_details(self, reference, hypothesis, **kwargs):
        return super(SegmentationRecall, self)._get_details(hypothesis, reference)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
