from logging import getLogger

# reward values. maybe use different rewards for finding
# weapons / ammo / medikits based on current inventory?
default_reward_values = {
    'BASE_REWARD': 0.,
    'DISTANCE': 9e-5,
    'KILL': 10.,
    'DEATH': -10.,
    'SUICIDE': -10.,
    'MEDIKIT': 0.4,
    'ARMOR': 0.4,
    'INJURED': -0.4,
    'WEAPON': 0.5,
    'AMMO': 0.5,
    'USE_AMMO': -0.5,
}

# logger
logger = getLogger()


class RewardBuilder(object):

    def __init__(self, game, values=None):
        self.game = game
        self.values = dict(default_reward_values)
        if values is not None:
            for k, v in values.items():
                assert k in self.values
                self.values[k] = v
        self.reset()
        logger.info("Reward values:")
        logger.info(self.values)

    def distance(self, d):
        self._reward += self.values['DISTANCE'] * d

    def kill(self, n_kills):
        self._reward += self.values['KILL'] * n_kills

    def death(self):
        self._reward += self.values['DEATH']

    def suicide(self):
        self._reward += self.values['SUICIDE']

    def medikit(self, hp):
        self._reward += self.values['MEDIKIT']

    def armor(self):
        self._reward += self.values['ARMOR']

    def injured(self, hp):
        self._reward += self.values['INJURED']

    def weapon(self):
        self._reward += self.values['WEAPON']

    def ammo(self):
        self._reward += self.values['AMMO']

    def use_ammo(self):
        self._reward += self.values['USE_AMMO']

    def reset(self):
        self._reward = self.values['BASE_REWARD']

    @property
    def reward(self):
        return self._reward
