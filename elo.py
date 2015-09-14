import numpy as np
import pandas as pd


class Elo(object):

    def __init__(self, teams, **kwargs):
        self.set_parameters(**kwargs)
        self.ratings = self._initalise_ratings(teams)

    def set_parameters(self, mean_rating=1500, k=20, home_advantage=0):
        if mean_rating < 0:
            raise ValueError("Mean rating must be >= 0")
        self._mean_rating = mean_rating

        if k <= 0:
            raise ValueError("K parameter must be > 0")
        self._k = k

        if home_advantage < 0:
            raise ValueError("Home advantage must be > 0")
        self._home_advantage = home_advantage

    def _initalise_ratings(self, teams):
        inital_values = np.empty(teams.size)
        inital_values.fill(self._mean_rating)
        return pd.Series(data=inital_values, index=teams)

    def calculate_likelihood(self, rating_a, rating_b, mov_multiplier=1, bias=0):
        """ Compute the expected score for team A given
        team A's rating (r_a) and team B's rating (r_b)
        """
        if rating_a < 0 or rating_b < 0:
            raise ValueError("Ratings must be positive")

        diff = (rating_b - (rating_a + bias))
        likelihood = 1.0 / (1.0 + 10**(diff / 400.0))
        return likelihood

    def _marign_of_victory_multiplier(self, rating_a, rating_b, mov):
        return np.log(np.fabs(mov)+1) * (2.2/((rating_a-rating_b)*.001+2.2))

    def update_rating(self, rating, score, likelihood, mov_multiplier=1):
        """ Compute the new rating from team A given their old rating (r_a)
        their actual score (s_a) and there expected score (e_a)
        """
        if rating < 0 or score < 0 or likelihood < 0:
            raise ValueError("Parameters must be positive")
        if likelihood > 1:
            raise ValueError("likelihood must be between 0 < likelihood < 1")

        rating = rating + self._k * mov_multiplier * (score - likelihood)
        return int(rating)

    def predict(self, games):
        if any([not isinstance(el, tuple) for el in games]):
            games = [games]

        predictions = [self._predict_single_game(game) for game in games]
        df = pd.concat(predictions, axis=1).T
        df.columns = ['Team A', 'Elo Score', 'Win %', 'Team B', 'Elo Score', 'Win %', 'Predicted Winner']
        return df

    def _predict_single_game(self, game):
        home_team, away_team = game

        if home_team not in self.ratings:
            raise KeyError('Team %s is not found in the ratings' % home_team)
        if away_team not in self.ratings:
            raise KeyError('Team %s is not found in the ratings' % away_team)

        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]

        likelihood_home = self.calculate_likelihood(home_rating, away_rating,
                                                    bias=self._home_advantage)
        likelihood_away = self.calculate_likelihood(away_rating, home_rating,
                                                    bias=-self._home_advantage)

        winner = home_team if likelihood_home > likelihood_away else away_team
        return pd.Series([home_team, home_rating, likelihood_home, away_team, away_rating, likelihood_away, winner])

    def train(self, game):
        home_team, home_pts, away_team, away_pts = game
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]

        home_score, away_score = self._calculate_game_points(home_pts, away_pts)
        prediction = self.predict((home_team, away_team))

        home_likelihood = prediction.ix[:, 2][0]
        away_likelihood = prediction.ix[:, 5][0]

        mov = (home_pts-away_pts)
        mov_multiplier = self._marign_of_victory_multiplier(home_rating, away_rating, mov)
        self.ratings[home_team] = self.update_rating(home_rating, home_score, home_likelihood, mov_multiplier)
        self.ratings[away_team] = self.update_rating(away_rating, away_score, away_likelihood, mov_multiplier)

    def _calculate_game_points(self, home_pts, away_pts):
        if home_pts == away_pts:
            # game was a draw, no gain for either team.
            return 0.5, 0.5
        elif home_pts > away_pts:
            # home team beat away team
            return 1, 0
        elif home_pts < away_pts:
            # away team beat home team
            return 0, 1

    def revert_ratings_to_mean(self, reversion_weight):
        if reversion_weight < 0 or reversion_weight > 1:
            raise ValueError('Reversion weight must be 0 < w < 1')
        self.ratings = (reversion_weight * self.ratings) + ((1-reversion_weight) * self._mean_rating)
        self.ratings = self.ratings.astype(int)
