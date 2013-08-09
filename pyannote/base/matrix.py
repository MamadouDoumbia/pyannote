#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012-2013 Herve BREDIN (bredin@limsi.fr)

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

import numpy as np
import networkx as nx
import warnings
import pandas


class LabelMatrix(object):

    def __init__(self, data=None, rows=None, columns=None, dtype=None):
        super(LabelMatrix, self).__init__()

        self.df = pandas.DataFrame(
            data=data, dtype=dtype if dtype else np.float,
            index=rows, columns=columns)

    def __get_shape(self):
        return self.df.shape
    shape = property(fget=__get_shape)

    def __get_labels(self):
        return list(self.df.index), list(self.df.columns)
    labels = property(fget=__get_labels)
    """List of labels"""

    def __setitem__(self, key, value):
        row = key[0]
        column = key[1]
        self.df = self.df.set_value(row, column, value)

    def __getitem__(self, key):
        row = key[0]
        column = key[1]
        return self.df.loc[row, column]

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        rows = self.df.index
        columns = self.df.columns
        for row in rows:
            for column in columns:
                yield row, column, self.df.loc[row, column]

    def copy(self):
        copied = self.__class__()
        copied.df = self.df.copy()
        return copied

    def __neg__(self):
        negated = self.__class__()
        negated.df = -self.df
        return negated

    def __iadd__(self):
        raise NotImplementedError()
        # for ilabel, jlabel, value in other:
        #     try:
        #         self[ilabel, jlabel] += value
        #     except Exception, e:
        #         self[ilabel, jlabel] = value
        # return self

    def __add__(self, other):
        added = self.__class__()
        added.df = self.df + other.df
        return added

    def max(self):
        return self.df.max().max()

    def min(self):
        return self.df.min().min()

    def argmax(self, axis=None):
        # if matrix is empty, return empty dictionary
        if not self:
            return {}

        if axis == 0:
            return {col: row for (col, row) in self.df.idxmax(axis=axis).iteritems()}
        elif axis == 1:
            return {row: col for (row, col) in self.df.idxmax(axis=axis).iteritems()}
        else:
            raise NotImplementedError()
        # indices = self.M.argmax(axis=axis)
        # if axis == 0:
        #     return {self.__jlabels[j] : self.__ilabels[i]
        #             for j, i in enumerate(indices)}
        # elif axis == 1:
        #     return {self.__ilabels[i] : self.__jlabels[j]
        #             for i, j in enumerate(indices)}
        # else:
        #     i, j = np.unravel_index([indices], self.shape, order='C')
        #     i = i[0]
        #     j = j[0]
        #     return {self.__ilabels[i] : self.__jlabels[j]}

    def argmin(self, axis=None):
        return (-self).argmax(axis=axis)

    def __str__(self):
        return str(self.df)

    @classmethod
    def from_networkx_graph(cls, G, multigraph_weight=sum, weight='weight'):

        M = np.array(nx.to_numpy_matrix(G,
                                        weight=weight,
                                        multigraph_weight=multigraph_weight))

        nodes = G.nodes()

        return LabelMatrix(data=M, rows=nodes, columns=nodes)

    def to_networkx_graph(self, weight='weight', selfloop=True):
        """Graphical representation

        Parameters
        ----------
        weight : str, optional
            Name of edge attribute. Defaults to 'weight'.
        selfloop : bool, optional
            When True, self edges (weighted with self similarity value) are
            included in the generated graph. Defaults to True.

        Returns
        -------
        g : networkx.DiGraph

        """
        G = nx.DiGraph()
        for i, j, d in self:
            if not selfloop and i == j:
                continue
            G.add_edge(i, j, {weight: d})
        return G

    def to_latex(self):

        I, J = self.labels

        # tabular environment
        source = '\\begin{tabular}{l|%s}\n' % ('c'*len(J))

        # header
        for j in J:
            source += '& %s ' % j
        source += '\\\\\n'
        source += '\\hline\n'

        # table content
        for i in I:
            source += '%s ' % i
            for j in J:
                source += '& %s ' % self[i, j]
            source += '\\\\\n'

        # tabular environment
        source += '\\end{tabular}\n'

        return source



# #########################################################
# TODO
# #########################################################

    # def __get_T(self):
    #     return LabelMatrix(ilabels=self.__jlabels,
    #                        jlabels=self.__ilabels,
    #                        Mij=None if self.__M is None else self.__M.T,
    #                        dtype=self.__dtype,
    #                        default=self.__default)
    # T = property(fget=__get_T)
    # """Transposed matrix"""

    # def __nonzero__(self):
    #     n, m = self.shape
    #     return n*m != 0

    # def __get_M(self):
    #     return self.__M
    # def __set_M(self, M):
    #     if M.shape != self.shape:
    #         raise ValueError('shape mismatch: %s vs. %s' % (M.shape,
    #                                                         self.shape))
    #     if M.dtype != self.__dtype:
    #         raise ValueError('data type mismatch.')
    #     self.__M = M
    # M = property(fget=__get_M, fset=__set_M)
    # "Internal numpy array"


    # def __getitem__(self, key):
    #     """

    #     """

    #     if not isinstance(key, tuple) or len(key) != 2:
    #         raise KeyError('')

    #     # get list of actual labels in matrix
    #     I_lbls, J_lbls = self.labels

    #     # get requested labels
    #     i_lbl = key[0]
    #     j_lbl = key[1]

    #     # special case for M['one_existing_label', 'another_existing_label']
    #     # return the actual stored value

    #     if not isinstance(i_lbl, set) and \
    #        i_lbl != slice(None, None, None) and \
    #        not isinstance(j_lbl, set) and \
    #        j_lbl != slice(None, None, None):
    #        try:
    #            i = self.__label2i[i_lbl]
    #            j = self.__label2j[j_lbl]
    #            return self.__M[i, j]
    #        except:
    #            raise KeyError('cannot get element [%s, %s]' % (i_lbl, j_lbl))

    #     # M[{'Bernard', 'John', 'Albert'}, ... ]
    #     if isinstance(i_lbl, set):
    #         i_lbls = sorted(i_lbl)
    #     # M[ :, ... ]
    #     elif i_lbl == slice(None, None, None):
    #         i_lbls = sorted(I_lbls)
    #     # M[ 'Bernard', ... ]
    #     else:
    #         i_lbls = [i_lbl]

    #     # M[ ..., {'Bernard', 'John', 'Albert'}]
    #     if isinstance(j_lbl, set):
    #         j_lbls = sorted(j_lbl)
    #     # M[ ..., : ]
    #     elif j_lbl == slice(None, None, None):
    #         j_lbls = sorted(J_lbls)
    #     # M[ ..., 'Bernard']
    #     else:
    #         j_lbls = [j_lbl]

    #     M = self.empty()
    #     try:
    #         for i_lbl in i_lbls:
    #             i = self.__label2i[i_lbl]
    #             for j_lbl in j_lbls:
    #                 j = self.__label2j[j_lbl]
    #                 M[i_lbl, j_lbl] = self.__M[i, j]
    #     except:
    #         raise KeyError('cannot get element [%s, %s]' % (i_lbl, j_lbl))

    #     return M

    # def __add_ilabel(self, ilabel):

    #     # get current shape (before adding label)
    #     n, m = self.__M.shape

    #     # append label at the end of list
    #     self.__ilabels.append(ilabel)
    #     # store its position
    #     self.__label2i[ilabel] = n

    #     # append default row to internal array
    #     default_row = np.empty((1, m), dtype=self.__dtype)
    #     default_row.fill(self.__default)
    #     self.__M = np.append(self.__M, default_row, axis=0)

    # def __add_jlabel(self, jlabel):

    #     # get current shape (before adding label)
    #     n, m = self.shape

    #     # append label at the end of list
    #     self.__jlabels.append(jlabel)
    #     # store its position
    #     self.__label2j[jlabel] = m

    #     # append default column to internal array
    #     default_column = np.empty((n, 1), dtype=self.__dtype)
    #     default_column.fill(self.__default)
    #     self.__M = np.append(self.__M, default_column, axis=1)

    # def __setitem__(self, key, value):
    #     """
    #     Values must be set one by one.

    #     Use expression 'matrix[label_i, label_j] = value

    #     """

    #     if not isinstance(key, tuple) or len(key) != 2:
    #         raise KeyError('')

    #     # get list of actual labels in matrix
    #     I_lbls, J_lbls = self.labels

    #     # get requested labels
    #     i_lbl = key[0]
    #     j_lbl = key[1]

    #     if isinstance(i_lbl, set) or i_lbl == slice(None, None, None) or \
    #        isinstance(j_lbl, set) or j_lbl == slice(None, None, None):
    #        raise KeyError('Cannot set multiple values at once.')

    #     if self.__M is None:
    #         self.__ilabels.append(i_lbl)
    #         self.__jlabels.append(j_lbl)
    #         self.__label2i[i_lbl] = 0
    #         self.__label2j[j_lbl] = 0
    #         self.__M = np.empty((1, 1), dtype=self.__dtype)
    #         self.__M[0, 0] = value
    #     else:
    #         if i_lbl not in self.__label2i:
    #             self.__add_ilabel(i_lbl)
    #         if j_lbl not in self.__label2j:
    #             self.__add_jlabel(j_lbl)
    #         i = self.__label2i[i_lbl]
    #         j = self.__label2j[j_lbl]
    #         self.__M[i, j] = value

    # def _delete_ilabel(self, ilabel):

    #     # find position in list
    #     i = self.__ilabels.index(ilabel)
    #     # remove from list
    #     self.__ilabels.remove(ilabel)
    #     # update label to position dictionary
    #     for i_lbl in self.__ilabels:
    #         if self.__label2i[i_lbl] > i:
    #             self.__label2i[i_lbl] -= 1
    #     # remove corresponding row from internal matrix
    #     self.__M = np.delete(self.__M, i, axis=0)

    # def _delete_jlabel(self, jlabel):

    #     # find position in list
    #     j = self.__jlabels.index(jlabel)
    #     # remove from list
    #     self.__jlabels.remove(jlabel)
    #     # update label to position dictionary
    #     for j_lbl in self.__jlabels:
    #         if self.__label2j[j_lbl] > j:
    #             self.__label2j[j_lbl] -= 1
    #     # remove corresponding row from internal matrix
    #     self.__M = np.delete(self.__M, j, axis=1)

    # def __delitem__(self, key):
    #     """Remove labels from matrix

    #     Examples
    #     --------

    #         Remove one label from the first set

    #         >>> del M[label, :]

    #         Remove one label from the second set

    #         >>> del M[:, label]

    #         Remove multiple labels at once

    #         >>> del M[set([label1, label2]), :]

    #     """
    #     if not isinstance(key, tuple) or len(key) != 2:
    #        raise KeyError('expecting: del M[ _ , :] or del M[:, _ ]')

    #     i_lbl = key[0]
    #     j_lbl = key[1]

    #     if i_lbl != slice(None, None, None) and \
    #        j_lbl != slice(None, None, None):
    #        raise KeyError('expecting: del M[ _ , :] or del M[:, _ ]')

    #     # get list of actual labels in matrix
    #     I_lbls, J_lbls = self.labels

    #     if i_lbl == slice(None, None, None):

    #         # M[ ..., {'Bernard', 'John', 'Albert'}]
    #         if isinstance(j_lbl, set):
    #             j_lbls = sorted(j_lbl)
    #         # M[ ..., : ]
    #         elif j_lbl == slice(None, None, None):
    #             j_lbls = sorted(J_lbls)
    #         # M[ ..., 'Bernard']
    #         else:
    #             j_lbls = [j_lbl]

    #         for j_lbl in j_lbls:
    #             self._delete_jlabel(j_lbl)

    #         return

    #     elif j_lbl == slice(None, None, None):

    #         # M[{'Bernard', 'John', 'Albert'}, ... ]
    #         if isinstance(i_lbl, set):
    #             i_lbls = sorted(i_lbl)
    #         # M[ :, ... ]
    #         elif i_lbl == slice(None, None, None):
    #             i_lbls = sorted(I_lbls)
    #         # M[ 'Bernard', ... ]
    #         else:
    #             i_lbls = [i_lbl]

    #         for i_lbl in i_lbls:
    #             self._delete_ilabel(i_lbl)

    #         return

    def iter_rows(self):
        return iter(self.df.index)

    def iter_columns(self):
        return iter(self.df)

    # def iter_ilabels(self):
    #     """
    #     First set label iterator
    #     """
    #     return iter(self.__ilabels)

    # def iter_jlabels(self):
    #     """
    #     Second set label iterator
    #     """
    #     return iter(self.__jlabels)

    # def __iter__(self):
    #     """
    #     (ilabel, jlabel, value) iterator
    #     """
    #     for ilabel in self.__ilabels:
    #         for jlabel in self.__jlabels:
    #             yield ilabel, jlabel, self[ilabel, jlabel]

    # def empty(self):
    #     """Empty copy"""
    #     return LabelMatrix(dtype=self.__dtype, default=self.__default)

    # def copy(self):
    #     """Duplicate matrix"""
    #     C = LabelMatrix(ilabels=list(self.__ilabels),
    #                     jlabels=list(self.__jlabels),
    #                     Mij=np.copy(self.__M), dtype=self.__dtype,
    #                     default=self.__default)
    #     return C

    # def __neg__(self):
    #     """Matrix opposite"""
    #     C = LabelMatrix(ilabels=list(self.__ilabels),
    #                     jlabels=list(self.__jlabels),
    #                     Mij=-np.copy(self.__M), dtype=self.__dtype,
    #                     default=-self.__default)
    #     return C

    # def __iadd__(self, other):
    #     """Add other matrix values"""
    #     for ilabel, jlabel, value in other:
    #         try:
    #             self[ilabel, jlabel] += value
    #         except Exception, e:
    #             self[ilabel, jlabel] = value
    #     return self

    # def __add__(self, other):
    #     """Add two matrices"""
    #     C = self.copy()
    #     return C.__iadd__(other)

    # def max(self):
    #     """Get matrix maximum value"""
    #     if not self:
    #         return np.nan

    #     return np.max(self.M)

    # def min(self):
    #     """Get matrix minimum value"""
    #     if not self:
    #         return np.nan

    #     return np.min(self.M)

    # def argmax(self, axis=None):

    #     # if matrix is empty, return empty dictionary
    #     if not self:
    #         return {}

    #     indices = self.M.argmax(axis=axis)
    #     if axis == 0:
    #         return {self.__jlabels[j] : self.__ilabels[i]
    #                 for j, i in enumerate(indices)}
    #     elif axis == 1:
    #         return {self.__ilabels[i] : self.__jlabels[j]
    #                 for i, j in enumerate(indices)}
    #     else:
    #         i, j = np.unravel_index([indices], self.shape, order='C')
    #         i = i[0]
    #         j = j[0]
    #         return {self.__ilabels[i] : self.__jlabels[j]}

    # def argmin(self, axis=None):
    #     return (-self).argmax(axis=axis)

    # def __str__(self):

    #     ilabels, jlabels = self.labels

    #     len_i = max([len(str(label)) for label in ilabels])
    #     len_j = max(4, max([len(str(label)) for label in jlabels]))

    #     fmt_label_i = "%%%ds" % len_i
    #     fmt_label_j = "%%%ds" % len_j
    #     fmt_value = "%%%d.2f" % len_j

    #     string = fmt_label_i % " "
    #     string += " "
    #     string += " ".join([fmt_label_j % str(j) for j in jlabels])

    #     for i in ilabels:
    #         string += "\n"
    #         string += fmt_label_i % str(i)
    #         string += " "
    #         string += " ".join([fmt_value % self[i, j] for j in jlabels])

    #     return string

    # def to_networkx_graph(self, weight='weight', selfloop=True):
    #     """Graphical representation

    #     Parameters
    #     ----------
    #     weight : str, optional
    #         Name of edge attribute. Defaults to 'weight'.
    #     selfloop : bool, optional
    #         When True, self edges (weighted with self similarity value) are
    #         included in the generated graph. Defaults to True.

    #     Returns
    #     -------
    #     g : networkx.DiGraph

    #     """
    #     G = nx.DiGraph()
    #     for i, j, d in self:
    #         if not selfloop and i==j:
    #             continue
    #         G.add_edge(i, j, {weight: d})
    #     return G

    # def to_latex(self):

    #     I, J = self.labels

    #     # tabular environment
    #     source = '\\begin{tabular}{l|%s}\n' % ('c'*len(J))

    #     # header
    #     for j in J:
    #         source += '& %s ' % j
    #     source += '\\\\\n'
    #     source += '\\hline\n'

    #     # table content
    #     for i in I:
    #         source += '%s ' % i
    #         for j in J:
    #             source += '& %s ' % self[i, j]
    #         source += '\\\\\n'

    #     # tabular environment
    #     source += '\\end{tabular}\n'

    #     return source

    # def _factorize(self, labels):
    #     import os.path
    #     tmp = [str(label) for label in labels]
    #     pmt = [label[::-1] for label in tmp]
    #     prefix = os.path.commonprefix(tmp)
    #     pre = len(prefix)
    #     suffix = os.path.commonprefix(pmt)[::-1]
    #     suf = len(suffix)
    #     if suf == 0:
    #         return prefix, [label[pre:] for label in tmp], suffix
    #     else:
    #         return prefix, [label[pre:-suf] for label in tmp], suffix




class Cooccurrence(LabelMatrix):
    """
    Cooccurrence matrix between two annotations

    Parameters
    ----------
    I, J : Annotation

    Returns
    --------
    matrix : Cooccurrence


    Examples
    --------

    >>> M = Cooccurrence(A, B)

    Get total confusion duration (in seconds) between id_A and id_B::

    >>> confusion = M[id_A, id_B]

    Get confusion dictionary for id_A::

    >>> confusions = M(id_A)

    """
    def __init__(self, I, J):

        # number of labels in annotations
        ni = len(I.labels())
        nj = len(J.labels())
        # initialize matrix with cooccurrence zero
        # Mij = np.zeros((ni, nj), dtype=float)
        super(Cooccurrence, self).__init__(rows=I.labels(),
                                           columns=J.labels(),
                                           dtype=np.float)

        for row in self.iter_rows():
            icov = I.label_coverage(row)
            for col in self.iter_columns():
                jcov = J.label_coverage(col)
                ijcov = icov.crop(jcov, mode='intersection').duration()
                self[row, col] = ijcov

from pyannote.base.segment import SEGMENT_PRECISION
class CoTFIDF(Cooccurrence):
    """Term Frequency Inverse Document Frequency (TF-IDF) confusion matrix

    C[i, j] = TF(i, j) x IDF(i) where
        - documents are J labels
        - words are co-occurring I labels

                  duration of word i in document j         confusion[i, j]
    TF(i, j) = --------------------------------------- = -------------------
               total duration of I words in document j   sum confusion[:, j]

                        number of J documents
    IDF(i) = ----------------------------------------------
             number of J documents co-occurring with word i

                      Nj
           = -----------------------
             sum confusion[i, :] > 0

    Parameters
    ---------
    words : :class:`pyannote.base.annotation.Annotation`
        Every label occurrence is considered a word
        (weighted by the duration of the segment)
    documents : :class:`pyannote.base.annotation.Annotation`
        Every label is considered a document.
    idf : bool, optional
        If `idf` is set to True, returns TF x IDF.
        Otherwise, returns TF. Default is True
    log : bool, optional
        If `log` is True, returns TF x log IDF

    """
    def __init__(self, words, documents, idf=True, log=False):

        # initialize as co-occurrence matrix
        super(CoTFIDF, self).__init__(words, documents)
        Nw, Nd = self.shape

        if Nd == 0:
            return

        # total duration of all words cooccurring with each document
        # np.sum(self.M, axis=0)[j] = 0 ==> self.M[i, j] = 0 for all i
        # so we can safely use np.maximum(1e-3, ...) to avoid DivideByZero
        tf = self.M / np.tile(np.maximum(SEGMENT_PRECISION,
                                         np.sum(self.M, axis=0)),
                              (Nw, 1))

        # use IDF only if requested (default is True ==> use IDF)
        if idf:

            # number of documents cooccurring with each word
            # np.sum(self.M > 0, axis=1)[i] = 0 ==> tf[i, j] = 0 for all i
            # and therefore tf.idf [i, j ] = 0
            # so we can safely use np.maximum(1, ...) to avoid DivideByZero
            idf = np.tile(float(Nd)/np.maximum(1, np.sum(self.M > 0, axis=1)), \
                          (Nd, 1)).T

            # use log only if requested (defaults is False ==> do not use log)
            if log:
                idf = np.log(idf)

        else:
            idf = 1.

        self.M = tf * idf


class AutoCooccurrence(Cooccurrence):
    """
    Auto confusion matrix

    Parameters
    ----------
    I : Annotation

    neighborhood : float, optional

    Examples
    --------

    >>> M = AutoCooccurrence(A, neighborhood=10)

    Get total confusion duration (in seconds) between id_A and id_B::

    >>> confusion = M[id_A, id_B]

    Get confusion dictionary for id_A::

    >>> confusions = M(id_A)


    """
    def __init__(self, I, neighborhood=0.):

        # extend each segment on left and right by neighborhood seconds
        segment_func = lambda s : neighborhood << s >> neighborhood
        xI = I.copy(segment_func=map_func)

        # auto-cooccurrence
        super(AutoCooccurrence, self).__init__(xI, xI)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

