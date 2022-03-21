from gym.envs.registration import register

register(
    id='csv-v0',
    entry_point='gym_csv.envs:CsvEnv',
)
register(
    id='csv-pygame-v0',
    entry_point='gym_csv.envs:CsvPyGameEnv',
)
