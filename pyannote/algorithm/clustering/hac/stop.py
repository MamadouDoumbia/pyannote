#!/usr/bin/env python
# encoding: utf-8

# Copyright 2013 Herve BREDIN (bredin@limsi.fr)

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


class HACStop(object):

    """Stopping criterion for hierarchical agglomerative clustering"""

    def __init__(self):
        super(HACStop, self).__init__()

    def initialize(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """
        Parameters
        ----------
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        """

        raise NotImplementedError("Method 'initialize' must be overriden.")

    def update(
        self, merged_clusters, new_cluster,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """

        Parameters
        ----------
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature
        """

        raise NotImplementedError("Method 'update' must be overriden.")

    def reached(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """

        Parameters
        ----------
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        """

        raise NotImplementedError("Method 'reached' must be overriden.")

    def finalize(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """

        Parameters
        ----------
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Returns
        -------
        final : Annotation
            Annotation when stop criterion is reached

        """

        raise NotImplementedError("Method 'is_reached' must be overriden.")
