import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
import collections
from nose.tools import *

from elo import Elo


class EloTest(unittest.TestCase):

    def setUp(self):
        self._teams = pd.Series(data=['TeamA', 'TeamB', 'TeamC', 'TeamD'])
        self._MEAN_ELO_RATING = 1500
        self._elo = Elo(self._teams, mean_rating=self._MEAN_ELO_RATING)

    def test_initialise_ratings(self):
        ratings = self._elo.ratings

        expected = np.empty(self._teams.shape)
        expected.fill(self._MEAN_ELO_RATING)

        npt.assert_equal(ratings.values, expected,
                         err_msg='All ratings should be equal on initalisation.')

    def test_calculate_likelihood(self):
        score = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, self._MEAN_ELO_RATING, 0)
        assert_equal(score, 0.5, 'Score should be 0.5.')

        score = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, 0, bias=100)
        assert_almost_equal(score, 1, places=3, msg='Score should almost be 1')

        score = self._elo.calculate_likelihood(0, self._MEAN_ELO_RATING, bias=100)
        assert_almost_equal(score, 0, places=3, msg='Score should almost be 0')

        score = self._elo.calculate_likelihood(0, self._MEAN_ELO_RATING, bias=100)
        assert_almost_equal(score, 0, places=3, msg='Score should almost be 0')

    def test_calculate_likelihood_with_home_advantage(self):
        score = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, 0, bias=100)
        assert_almost_equal(score, 1, places=3, msg='Score should almost be 1')

        score = self._elo.calculate_likelihood(0, self._MEAN_ELO_RATING, bias=-100)
        assert_almost_equal(score, 0, places=3, msg='Score should almost be 0')

        score = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, 1, bias=100)
        score_unbiased = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, 1, 100, bias=-100)
        assert_not_equal(score, score_unbiased, msg='Score should not exactly equal, but should be close.')

        score = self._elo.calculate_likelihood(self._MEAN_ELO_RATING, self._MEAN_ELO_RATING, bias=100)
        print score
        assert_almost_equal(score, 0.640, places=3, msg='Score should almost be biased towards team A.')

    def test_calculate_likelihood_invalid(self):
        assert_raises(ValueError, self._elo.calculate_likelihood, self._MEAN_ELO_RATING, -1, bias=100)
        assert_raises(ValueError, self._elo.calculate_likelihood, -1, self._MEAN_ELO_RATING, bias=100)
        assert_raises(ValueError, self._elo.calculate_likelihood, -1, -1, bias=100)

    def test_update_rating(self):
        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 1, 0.5)
        assert_equal(rating, 1510, 'Rating should be increased by 10.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 0, 0.5)
        assert_equal(rating, 1490, 'Rating should be decreased by 10.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 0.5, 0.25)
        assert_equal(rating, 1505, 'Rating should increase by 5.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 0.5, 0.75)
        assert_equal(rating, 1495, 'Rating should decrease by 5.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 1, 1.0)
        assert_equal(rating, 1500, 'Rating should be not change, we expected a win.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 0, 0.0)
        assert_equal(rating, 1500, 'Rating should be not change, we expected a loss.')

        rating = self._elo.update_rating(self._MEAN_ELO_RATING, 0.5, 0.5)
        assert_equal(rating, 1500, 'Rating should not change, we expected a tie.')

    def test_update_rating_invalid(self):
        assert_raises(ValueError, self._elo.update_rating, -1, 1, 0.5)
        assert_raises(ValueError, self._elo.update_rating, 1, -1, 0.5)
        assert_raises(ValueError, self._elo.update_rating, 1, 1, -0.5)

    def test_predict_equal_ratings(self):
        game = ('TeamA', 'TeamB')
        prediction = self._elo.predict(game)
        team_a_likelihood = prediction.ix[:, 2][0]
        team_b_likelihood = prediction.ix[:, 5][0]
        assert_equal(team_a_likelihood, team_b_likelihood,
                     'Teams with equal ratings should be equally likely')

    def test_predict_unequal_ratings(self):
        self._elo.ratings['TeamA'] = 1800
        self._elo.ratings['TeamB'] = 1200

        game = ('TeamA', 'TeamB')
        prediction = self._elo.predict(game)
        team_a_likelihood = prediction.ix[:, 2][0]
        team_b_likelihood = prediction.ix[:, 5][0]

        assert_almost_equal(team_a_likelihood, 0.969346569968,
                            msg='Team A should be more likely to win')
        assert_almost_equal(team_b_likelihood, 0.0306534300317,
                            msg='Teams B should be less likely to win')
        assert_almost_equal(team_a_likelihood+team_b_likelihood, 1, places=2,
                     msg='Total probability should sum to 1.')

    def test_predict_key_error(self):
        assert_raises(KeyError, self._elo.predict, ('NotATeam', 'TeamB'))
        assert_raises(KeyError, self._elo.predict, ('TeamA', 'NotATeam'))
        assert_raises(KeyError, self._elo.predict, ('NotATeam', 'NotATeam'))

    def test_predict_vectorised(self):
        test_data = [('TeamA', 'TeamB'), ('TeamC', 'TeamD')]
        predictions = self._elo.predict(test_data)

        assert_true(isinstance(predictions, pd.DataFrame))
        assert_equal(predictions.shape, (2, 7))

        team_a_likelihood = predictions.ix[:, 2][0]
        team_b_likelihood = predictions.ix[:, 5][0]
        assert_almost_equal(team_a_likelihood+team_b_likelihood, 1, places=2,
                     msg='Total probability should sum to 1.')

        team_c_likelihood = predictions.ix[:, 2][1]
        team_d_likelihood = predictions.ix[:, 5][1]
        assert_almost_equal(team_c_likelihood+team_d_likelihood, 1, places=2,
                     msg='Total probability should sum to 1.')

    def test_train_unequal(self):
        game = ('TeamA', 23, 'TeamB', 14)
        self._elo.train(game)

        assert_equal(self._elo.ratings['TeamA'], 1523.0, 'TeamA should of increased their rating.')
        assert_equal(self._elo.ratings['TeamB'], 1476.0, 'TeamB should of decreased their rating.')

    def test_train_equal(self):
        game = ('TeamA', 10, 'TeamB', 10)
        self._elo.train(game)

        assert_equal(self._elo.ratings['TeamA'], self._elo.ratings['TeamB'],
                     'Game was a draw. Ratings should remain the same.')

    def test_train_with_home_advantage(self):
        self._elo._home_advantage = 100

        game = ('TeamA', 10, 'TeamB', 10)
        self._elo.train(game)

        assert_equal(self._elo.ratings['TeamA'], self._elo.ratings['TeamB'],
                         'Home team advantage should cause a draw not to result in equal rankings.')

        game = ('TeamA', 11, 'TeamB', 10)
        self._elo.train(game)

        assert_not_equal(self._elo.ratings['TeamA'], self._elo.ratings['TeamB'],
                         'Home team advantage should cause a draw not to result in equal rankings.')
        assert_almost_equal(self._elo.ratings['TeamA'], 1504.0,
                            msg='Team A increases slightly.')
        assert_almost_equal(self._elo.ratings['TeamB'], 1495.0,
                            msg='Team B decreases slightly.')

    def test_train_key_error(self):
        assert_raises(KeyError, self._elo.train, ('NotATeam', 1, 'TeamB', 1))
        assert_raises(KeyError, self._elo.train, ('TeamA', 1, 'NotATeam', 1))
        assert_raises(KeyError, self._elo.train, ('NotATeam', 1, 'NotATeam', 1))

    def test_revert_ratings_to_mean(self):
        self._elo.ratings['TeamA'] = 1800
        self._elo.ratings['TeamB'] = 1200

        self._elo.revert_ratings_to_mean(0.25)
        team_a = self._elo.ratings.ix[0]
        team_b = self._elo.ratings.ix[1]

        assert_equal(team_a, 1575.0, 'Rating should be reverted to within 25\% of the mean.')
        assert_equal(team_b, 1425.0, 'Rating should be reverted to within 25\% of the mean.')

    def test_revert_ratings_to_mean_invalid(self):
        assert_raises(ValueError, self._elo.revert_ratings_to_mean, -0.75)


if __name__ == "__main__":
    unittest.main()
