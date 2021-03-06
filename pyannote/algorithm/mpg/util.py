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

import networkx as nx
import numpy as np
from pyannote.algorithm.mpg.node import IdentityNode, LabelNode, TrackNode
from pyannote.algorithm.mpg.graph import RANK, PROBABILITY, SUBTRACK, COOCCURRING

def remove_nbest_identity(G, nbest):
    """Remove any identity nodes not in any n-best list"""
    inodes = [node for node in G if isinstance(node, IdentityNode)]
    remove = []
    for n, inode in enumerate(inodes):
        ranks = set([G[inode][node].get(RANK, np.inf)
                     for node in G.neighbors_iter(inode)])
        if all([r>nbest for r in ranks]):
            remove.append(inode)
    G.remove_nodes_from(remove)
    return G


def meta_mpg(g):
    """Meta Multimodal Probability Graph

    Parameters
    ----------
    g : nx.Graph
        Multimodal probability graph

    Returns
    -------
    G : nx.Graph
        Multimodal probability graph where hard-linked nodes (p=1) are
        grouped into meta-nodes
    groups : list of lists
        Groups of nodes
    """

    # Group of hard-linked nodes
    # (ie. nodes connected with probability p=1)
    hard = nx.Graph()
    hard.add_nodes_from(g)
    hard.add_edges_from([(e,f) for e,f,d in g.edges_iter(data=True)
                               if d[PROBABILITY] == 1.])
    groups = nx.connected_components(hard)

    # meta graph with one node per group
    G = nx.blockmodel(g, groups, multigraph=True)

    meta = nx.Graph()
    for n in range(len(groups)):
        meta.add_node(n)
        for m in range(n+1, len(groups)):

            # do not do anything in case there is no edge
            # between those two meta-nodes
            if not G.has_edge(n, m):
                continue

            # obtain probabilities of all edges between n & m
            probabilities = [data[PROBABILITY] for data in G[n][m].values()]

            # raise an error in case of conflict (p=0 vs. p>0)
            if len(set(probabilities)) > 1 and 0 in probabilities:
                raise ValueError('conflict in meta-edges between %r and %r:' \
                                 'probabilities = %r' % (groups[n],
                                                         groups[m],
                                                         probabilities))

            meta.add_edge(n, m, {PROBABILITY: np.mean(probabilities)})

    return meta, groups


def log_mpg(g):
    """Make log-probability graph from probability graph

    Parameters
    ----------
    g: nx.Graph
        Probability graph

    Returns
    -------
    log : nx.Graph
        Input graph where each edge probability P is replaced by -log P
        except when P = 0 (otherwise -log P = +oo and subsequent shortest
        path algorithm fails quietly)

    """

    # new graph containing nodes from input graph
    log = nx.Graph()
    log.add_nodes_from(g.nodes_iter(data=True))

    # convert P to -log P for all input edges
    # do not add edge when P = 0
    for e,f,d in g.edges_iter(data=True):
        D = dict(d)
        p = d[PROBABILITY]
        if p > 0:
            D[PROBABILITY] = -np.log(p)
            log.add_edge(e,f,D)
    return log


def complete_mpg(g):

    G = propagate_constraints(g)

    log = log_mpg(G)
    complete = nx.Graph()

    # all track nodes of interest
    # ie speaker/head node not subtrack
    tnodes = [(n,d) for n,d in G.nodes_iter(data=True) \
                    if isinstance(n, TrackNode) \
                    and n.modality in ['speaker', 'head'] \
                    and not d.get(SUBTRACK, False)]

    # all identity nodes
    inodes = [(n,d) for n,d in G.nodes_iter(data=True) \
                    if isinstance(n, IdentityNode)]

    # tnode/tnode shortest path (with forbidden identity nodes)
    _log = nx.Graph(log)
    _log.remove_nodes_from(zip(*inodes)[0])
    _shortest = nx.shortest_path_length(_log, weight=PROBABILITY)
    for i, (n, d) in enumerate(tnodes):
        complete.add_node(n, **d)
        for N, D in tnodes[i+1:]:
            if G.has_edge(n, N):
                data = dict(G[n][N])
            else:
                data = {PROBABILITY: np.exp(-_shortest[n][N])}
            complete.add_edge(n, N, **data)

    # inode/tnodes shortest path (with forbidden other identity nodes)
    for i, (n, d) in enumerate(inodes):
        complete.add_node(n, **d)
        _log = nx.Graph(log)
        _log.remove_nodes_from([m for j,(m,_) in enumerate(inodes) if j != i])
        _shortest = nx.shortest_path_length(_log, source=n, weight=PROBABILITY)
        for N, D in tnodes:
            if G.has_edge(n, N):
                data = dict(G[n][N])
            else:
                data = {PROBABILITY: np.exp(-_shortest[N])}
            complete.add_edge(n, N, **data)

    # inode/inode constraint
    for i, (n, d) in enumerate(inodes):
        for m,_ in inodes[i+1:]:
            G.add_edge(n, m, **{PROBABILITY: 0})

    return complete


def densify(g, copy=True):
    """
    Densify multimodal probability graph

    Add missing "track to identity" edges using maximum probability path

    Parameters
    ----------
    g : `MultimodalProbabilityGraph`
    copy : bool, optional
        When `True`, returns a densified copy of `g`.
        When `False`, densify `g` itself (in place).
    """

    # create dual directed graph
    # * only outgoing edges for Identity nodes
    # * weight = -log probability
    dual = nx.DiGraph()
    dual.add_nodes_from(g)
    for e, f, d in g.edges_iter(data=True):
        weight = -np.log(d[PROBABILITY])
        if isinstance(e, IdentityNode):
            if not isinstance(f, IdentityNode):
                dual.add_edge(e, f, weight=weight)
        else:
            dual.add_edge(f, e, weight=weight)
            if not isinstance(f, IdentityNode):
                dual.add_edge(e, f, weight=weight)

    # get list of tracks and identities
    tnodes = g.tnodes()
    inodes = g.inodes()

    # duplicate g if requested
    if copy:
        g = g.copy()

    # add missing edges via maximum probability path
    for inode in inodes:
        weights = {}
        for tnode in tnodes:
            if not g.has_edge(inode, tnode):

                # compute maximum probability paths from current identity
                # only if necessary (iff it was not already computed
                # and there is a missing edge)
                if not weights:
                    weights = nx.shortest_path_length(dual, source=inode, weight='weight')

                # if there exists at least one path from identity to track
                # use the maximum probability path
                if tnode in weights:
                    g.update_edge(inode, tnode,
                                  probability=np.exp(-weights[tnode]))

    return g

