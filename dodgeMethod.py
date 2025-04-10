from mark_Runner import BenchmarkRunner
from bot_ai import GameBot
from game import Game 
from settings import DodgeMethod

game = Game()
DodgeMethod = {
    "Furthest Safe Direction": lambda: GameBot(game, DodgeMethod.FURTHEST_SAFE_DIRECTION),
    "Least Danger": lambda: GameBot(game, DodgeMethod.LEAST_DANGER_PATH),
    "Least Danger Advanced": lambda: GameBot(game, DodgeMethod.LEAST_DANGER_PATH_ADVANCED),
    "Opposite Threat Direction": lambda: GameBot(game, DodgeMethod.OPPOSITE_THREAT_DIRECTION),
    "Random Safe Zone": lambda: GameBot(game, DodgeMethod.RANDOM_SAFE_ZONE),
}
runner = BenchmarkRunner
save_path= "/content/drive/MyDrive/game_ai/benchmark_results.csv"
runner.run(DodgeMethod,save_csv=True,csv = save_path)
