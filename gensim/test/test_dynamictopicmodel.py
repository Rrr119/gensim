#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Tests for :class:`~gensim.models.dynamictopicmodel.DynamicTopicModel`."""

import logging
import unittest

import numpy as np

from gensim.models import DynamicTopicModel
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile


class TestDynamicTopicModel(unittest.TestCase):
    def setUp(self):
        self.time_slice = [3, 3, 3]
        self.num_topics = 3
        self.model = DynamicTopicModel(
            corpus=common_corpus, id2word=common_dictionary,
            time_slice=self.time_slice, num_topics=self.num_topics,
            passes=5, random_state=42,
        )

    def test_trains_one_model_per_slice(self):
        self.assertEqual(self.model.num_slices, len(self.time_slice))
        self.assertEqual(len(self.model.slice_models), len(self.time_slice))
        for slice_model in self.model.slice_models:
            self.assertEqual(slice_model.num_topics, self.num_topics)
        for slice_corpus, slice_size in zip(self.model.slice_corpora, self.time_slice):
            self.assertEqual(len(slice_corpus), slice_size)

    def test_alignment_is_permutation(self):
        self.assertEqual(len(self.model.alignments), len(self.time_slice))
        for alignment in self.model.alignments:
            self.assertEqual(sorted(alignment.tolist()), list(range(self.num_topics)))
        # the first slice defines the canonical topic order
        self.assertEqual(self.model.alignments[0].tolist(), list(range(self.num_topics)))

    def test_alignment_maximizes_similarity(self):
        # the aligned similarity must be at least as high as the identity (unaligned) one
        reference = self.model.slice_models[0].get_topics()
        for time, slice_model in enumerate(self.model.slice_models[1:], start=1):
            topics = slice_model.get_topics()
            similarity = DynamicTopicModel._cosine_similarity(reference, topics)
            aligned = similarity[np.arange(self.num_topics), self.model.alignments[time]].sum()
            identity = np.diag(similarity).sum()
            self.assertGreaterEqual(aligned, identity - 1e-10)
            reference = topics[self.model.alignments[time]]

    def test_topic_evolution(self):
        topn = 5
        for topic_id in range(self.num_topics):
            evolution = self.model.topic_evolution(topic_id, topn=topn)
            self.assertEqual([time for time, _ in evolution], list(range(len(self.time_slice))))
            for _, terms in evolution:
                self.assertEqual(len(terms), topn)
                for term, probability in terms:
                    self.assertIsInstance(term, str)
                    self.assertGreaterEqual(probability, 0.0)
                    self.assertLessEqual(probability, 1.0)

    def test_print_topic_evolution(self):
        lines = self.model.print_topic_evolution(topic_id=0, topn=4)
        self.assertEqual(len(lines), len(self.time_slice))
        for time, line in enumerate(lines):
            self.assertIn("time=%d" % time, line)
            self.assertIn("*", line)

    def test_topic_proportions_evolution(self):
        proportions = self.model.topic_proportions_evolution()
        self.assertEqual(proportions.shape, (self.num_topics, len(self.time_slice)))
        np.testing.assert_allclose(proportions.sum(axis=0), np.ones(len(self.time_slice)), atol=1e-6)
        self.assertTrue(np.all(proportions >= 0.0))

    def test_get_document_topics(self):
        bow = common_corpus[0]
        for time in range(len(self.time_slice)):
            distribution = self.model.get_document_topics(bow, time=time)
            self.assertEqual(len(distribution), self.num_topics)
            topic_ids = [topic_id for topic_id, _ in distribution]
            self.assertEqual(sorted(topic_ids), list(range(self.num_topics)))
            total = sum(probability for _, probability in distribution)
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_get_document_topics_minimum_probability(self):
        bow = common_corpus[0]
        distribution = self.model.get_document_topics(bow, time=0, minimum_probability=0.5)
        self.assertTrue(all(probability >= 0.5 for _, probability in distribution))

    def test_reproducible_with_random_state(self):
        other = DynamicTopicModel(
            corpus=common_corpus, id2word=common_dictionary,
            time_slice=self.time_slice, num_topics=self.num_topics,
            passes=5, random_state=42,
        )
        for first, second in zip(self.model.slice_models, other.slice_models):
            np.testing.assert_allclose(first.get_topics(), second.get_topics())

    def test_save_load(self):
        fname = get_tmpfile("dtm.model")
        self.model.save(fname)
        loaded = DynamicTopicModel.load(fname)
        self.assertEqual(loaded.num_slices, self.model.num_slices)
        self.assertEqual(loaded.time_slice, self.model.time_slice)
        for alignment, loaded_alignment in zip(self.model.alignments, loaded.alignments):
            np.testing.assert_array_equal(alignment, loaded_alignment)
        evolution = self.model.topic_evolution(0, topn=5)
        loaded_evolution = loaded.topic_evolution(0, topn=5)
        self.assertEqual(len(evolution), len(loaded_evolution))
        for (_, terms), (_, loaded_terms) in zip(evolution, loaded_evolution):
            self.assertEqual([term for term, _ in terms], [term for term, _ in loaded_terms])

    def test_time_slice_must_match_corpus_size(self):
        with self.assertRaises(ValueError):
            DynamicTopicModel(
                corpus=common_corpus, id2word=common_dictionary,
                time_slice=[2, 2], num_topics=2,
            )

    def test_time_slice_must_be_positive(self):
        with self.assertRaises(ValueError):
            DynamicTopicModel(
                corpus=common_corpus, id2word=common_dictionary,
                time_slice=[9, 0], num_topics=2,
            )

    def test_missing_time_slice_raises(self):
        with self.assertRaises(ValueError):
            DynamicTopicModel(corpus=common_corpus, id2word=common_dictionary, num_topics=2)

    def test_invalid_topic_or_time(self):
        with self.assertRaises(ValueError):
            self.model.topic_evolution(topic_id=self.num_topics)
        with self.assertRaises(ValueError):
            self.model.get_document_topics(common_corpus[0], time=len(self.time_slice))

    def test_unaligned_model_uses_identity(self):
        model = DynamicTopicModel(
            corpus=common_corpus, id2word=common_dictionary,
            time_slice=self.time_slice, num_topics=self.num_topics,
            passes=1, random_state=42, align=False,
        )
        for alignment in model.alignments:
            self.assertEqual(alignment.tolist(), list(range(self.num_topics)))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG
    )
    unittest.main()
