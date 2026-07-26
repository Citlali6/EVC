import math
import unittest

from utils.challenge_eval import SCORE_FA_SCALE, challenge_score


class ChallengeScoreTests(unittest.TestCase):
    def test_score_without_false_alarms(self):
        score_fa, score = challenge_score(iou=0.4, acc=0.6, pd=0.8, fa=0.0)

        self.assertEqual(score_fa, 1.0)
        self.assertAlmostEqual(score, 0.76)

    def test_false_alarm_term_uses_official_scale(self):
        score_fa, score = challenge_score(iou=0.0, acc=0.0, pd=0.0, fa=1e-5)

        self.assertAlmostEqual(score_fa, math.exp(-SCORE_FA_SCALE * 1e-5))
        self.assertAlmostEqual(score, 0.3 * score_fa)


if __name__ == '__main__':
    unittest.main()
