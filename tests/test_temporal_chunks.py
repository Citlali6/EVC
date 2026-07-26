import unittest

from dataset.temporal_chunks import partition_event_indices


class TemporalChunkTests(unittest.TestCase):
    def test_empty_input_has_no_chunks(self):
        self.assertEqual(partition_event_indices(0, 100), [])

    def test_exact_budget_stays_one_chunk(self):
        self.assertEqual(partition_event_indices(100, 100), [(0, 100)])

    def test_chunks_cover_every_event_without_overlap(self):
        chunks = partition_event_indices(250001, 100000)
        self.assertEqual(
            chunks,
            [(0, 100000), (100000, 200000), (200000, 250001)],
        )
        self.assertEqual(sum(end - start for start, end in chunks), 250001)
        self.assertTrue(all(end - start <= 100000 for start, end in chunks))

    def test_invalid_arguments_fail_early(self):
        with self.assertRaises(ValueError):
            partition_event_indices(-1, 100)
        with self.assertRaises(ValueError):
            partition_event_indices(1, 0)


if __name__ == '__main__':
    unittest.main()
