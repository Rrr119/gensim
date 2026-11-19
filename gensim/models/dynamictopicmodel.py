#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Dynamic Topic Model via per-slice LDA with topic alignment.

A lightweight alternative to :class:`~gensim.models.ldaseqmodel.LdaSeqModel`.
Instead of a full state-space model, an independent
:class:`~gensim.models.ldamodel.LdaModel` is trained on each time slice, and
the topics of consecutive slices are then aligned (matched) by solving a
linear assignment problem on the cosine similarities of their topic-word
distributions. After alignment, topic id ``k`` refers to the "same" topic in
every slice, which makes it possible to track how each topic evolves over
time.

Examples
--------
Train a dynamic topic model on 9 documents split into 3 time slices:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.models import DynamicTopicModel
    >>>
    >>> dtm = DynamicTopicModel(
    ...     corpus=common_corpus, id2word=common_dictionary,
    ...     time_slice=[3, 3, 3], num_topics=2, random_state=42,
    ... )

Inspect how the top terms of topic 0 evolve over the slices:

.. sourcecode:: pycon

    >>> evolution = dtm.topic_evolution(topic_id=0, topn=5)
    >>> for time, terms in evolution:
    ...     print(time, [term for term, _ in terms])

Get the average proportion of each topic in each time slice:

.. sourcecode:: pycon

    >>> proportions = dtm.topic_proportions_evolution()

Persist the model to disk:

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>>
    >>> fname = get_tmpfile("dtm.model")
    >>> dtm.save(fname)
    >>> dtm = DynamicTopicModel.load(fname)

"""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from gensim import utils
from gensim.models.ldamodel import LdaModel

logger = logging.getLogger(__name__)


class DynamicTopicModel(utils.SaveLoad):
    """Sequence of per-slice LDA models with aligned topics."""

    def __init__(self, corpus=None, time_slice=None, id2word=None, num_topics=10,
                 align=True, random_state=None, **lda_kwargs):
        """
        Parameters
        ----------
        corpus : iterable of list of (int, int), optional
            Corpus in bag-of-words format. Documents must be ordered by time:
            the first ``time_slice[0]`` documents belong to the first slice, and so on.
        time_slice : list of int, optional
            Number of documents in each time slice. Must sum to ``len(corpus)``.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping from word ids to words, used to name the terms in the output.
        num_topics : int, optional
            Number of topics. Identical for every slice, so that topics can be aligned.
        align : bool, optional
            If True, align the topics of each slice with the previous slice by
            maximizing the total cosine similarity of matched topic-word
            distributions (Hungarian algorithm). If False, topics keep their
            raw per-slice ids.
        random_state : int or :class:`numpy.random.RandomState`, optional
            Seed for the per-slice LDA models, for reproducible results.
        **lda_kwargs
            Additional keyword arguments passed to each per-slice
            :class:`~gensim.models.ldamodel.LdaModel` (e.g. ``passes``, ``alpha``).

        """
        self.num_topics = num_topics
        self.id2word = id2word
        self.align = align
        self.time_slice = list(time_slice) if time_slice is not None else None
        self.lda_kwargs = lda_kwargs
        #: Per-slice LDA models, in temporal order.
        self.slice_models = []
        #: Per-slice corpora, in temporal order.
        self.slice_corpora = []
        #: ``alignments[t][global_id]`` = local topic id of ``global_id`` in slice ``t``.
        self.alignments = []

        if corpus is None:
            if time_slice is not None:
                raise ValueError("cannot initialize a model with time_slice but no corpus")
            return

        corpus = list(corpus)
        if self.time_slice is None:
            raise ValueError("time_slice must be provided together with a corpus")
        if sum(self.time_slice) != len(corpus):
            raise ValueError(
                "time_slice %s does not sum to the corpus size %d" % (self.time_slice, len(corpus))
            )
        if any(size <= 0 for size in self.time_slice):
            raise ValueError("each time slice must contain at least one document")

        if isinstance(random_state, np.random.RandomState):
            seed_sequence = random_state.randint(0, 2 ** 31 - 1, size=len(self.time_slice))
        elif random_state is None:
            seed_sequence = [None] * len(self.time_slice)
        else:
            seed_sequence = np.random.RandomState(random_state).randint(
                0, 2 ** 31 - 1, size=len(self.time_slice)
            )

        offset = 0
        for slice_index, slice_size in enumerate(self.time_slice):
            slice_corpus = corpus[offset:offset + slice_size]
            offset += slice_size
            logger.info("training LDA on time slice %d with %d documents", slice_index, slice_size)
            model = LdaModel(
                corpus=slice_corpus, id2word=id2word, num_topics=num_topics,
                random_state=seed_sequence[slice_index], **lda_kwargs
            )
            self.slice_models.append(model)
            self.slice_corpora.append(slice_corpus)

        self.alignments = self._align_topics() if align else self._identity_alignments()

    @property
    def num_slices(self):
        """Number of time slices in the model."""
        return len(self.slice_models)

    def _identity_alignments(self):
        identity = np.arange(self.num_topics)
        return [identity.copy() for _ in self.slice_models]

    def _align_topics(self):
        """Match topics of each slice to the previous slice via linear assignment."""
        alignments = [np.arange(self.num_topics)]
        reference = self.slice_models[0].get_topics()
        for model in self.slice_models[1:]:
            topics = model.get_topics()
            similarity = self._cosine_similarity(reference, topics)
            # Hungarian algorithm: assign each reference (global) topic to one local topic.
            _, local_ids = linear_sum_assignment(-similarity)
            permutation = np.asarray(local_ids)
            alignments.append(permutation)
            reference = topics[permutation]
        return alignments

    @staticmethod
    def _cosine_similarity(left, right):
        """Cosine similarity between each row of `left` and each row of `right`."""
        left_norm = left / np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = right / np.linalg.norm(right, axis=1, keepdims=True)
        return np.dot(left_norm, right_norm.T)

    def _local_topic_id(self, topic_id, time):
        """Translate a global (aligned) topic id to the local id used in slice `time`."""
        if not 0 <= topic_id < self.num_topics:
            raise ValueError("topic_id %r out of range [0, %d)" % (topic_id, self.num_topics))
        if not 0 <= time < self.num_slices:
            raise ValueError("time %r out of range [0, %d)" % (time, self.num_slices))
        return int(self.alignments[time][topic_id])

    def topic_evolution(self, topic_id, topn=10):
        """Get the top terms of a topic in every time slice.

        Parameters
        ----------
        topic_id : int
            Global (aligned) id of the topic to track.
        topn : int, optional
            Number of top terms to return per slice.

        Returns
        -------
        list of (int, list of (str, float))
            ``(time, [(term, probability), ...])`` for each time slice.

        """
        evolution = []
        for time, model in enumerate(self.slice_models):
            local_id = self._local_topic_id(topic_id, time)
            terms = [
                (model.id2word[word_id], probability)
                for word_id, probability in model.get_topic_terms(local_id, topn=topn)
            ]
            evolution.append((time, terms))
        return evolution

    def print_topic_evolution(self, topic_id, topn=10):
        """Format the evolution of a topic as printable strings.

        Parameters
        ----------
        topic_id : int
            Global (aligned) id of the topic to track.
        topn : int, optional
            Number of top terms per slice.

        Returns
        -------
        list of str
            One string per time slice, e.g. ``"time=0: 0.042*\\"cat\\" + ..."``.

        """
        lines = []
        for time, terms in self.topic_evolution(topic_id, topn=topn):
            rendered = " + ".join('%.3f*"%s"' % (probability, term) for term, probability in terms)
            lines.append("time=%d: %s" % (time, rendered))
        return lines

    def topic_proportions_evolution(self):
        """Compute the average proportion of each topic in each time slice.

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of shape ``(num_topics, num_slices)``; each column sums to 1
            and holds the mean document-topic distribution of that slice,
            expressed in global (aligned) topic ids.

        """
        proportions = np.zeros((self.num_topics, self.num_slices))
        for time, (model, slice_corpus) in enumerate(zip(self.slice_models, self.slice_corpora)):
            local_proportions = np.zeros(self.num_topics)
            for bow in slice_corpus:
                gamma = model.get_document_topics(bow, minimum_probability=0.0)
                local_proportions += np.asarray([probability for _, probability in gamma])
            local_proportions /= len(slice_corpus)
            for global_id in range(self.num_topics):
                local_id = self._local_topic_id(global_id, time)
                proportions[global_id, time] = local_proportions[local_id]
        return proportions

    def get_document_topics(self, bow, time, minimum_probability=0.0):
        """Infer the topic distribution of a document in a given time slice.

        Parameters
        ----------
        bow : list of (int, int)
            Document in bag-of-words format.
        time : int
            Index of the time slice whose LDA model should be used.
        minimum_probability : float, optional
            Topics with a lower probability are excluded from the result.

        Returns
        -------
        list of (int, float)
            ``(global_topic_id, probability)`` pairs.

        """
        if not 0 <= time < self.num_slices:
            raise ValueError("time %r out of range [0, %d)" % (time, self.num_slices))
        model = self.slice_models[time]
        local_distribution = model.get_document_topics(bow, minimum_probability=0.0)
        result = []
        for local_id, probability in local_distribution:
            global_id = int(np.flatnonzero(self.alignments[time] == local_id)[0])
            if probability >= minimum_probability:
                result.append((global_id, probability))
        return result
