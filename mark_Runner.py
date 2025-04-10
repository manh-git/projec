import time
import numpy as np
from game import Game
import csv

class BenchmarkRunner:
    def __init__(self,num_runs=10):
        self.num_runs = num_runs
        self.results = {}
    def run(self,dodgeMethod, save_csv = True, csv_filename="benchmark_result.csv"):
        all_data = []
        for name, dodgeMethod in dodgeMethod.items():
            print(f"Running:{name}")
            times= []
            for i in range(self.num_runs):
                bot = dodgeMethod()
                start = time.perf_counter()
                bot.update()
                end = time.perf_counter()
                duration = end -start
                times.append(duration)
                all_data.append({"algorithim": name, "run": i + 1,"time": duration})
            self.results[name]={
                'runs': times
            }
        if save_csv:
            keys =["algorithim","run","time"]
            with open(csv_filename,"w",newline='') as f:
                writer = csv.DictWriter(f, fieldnames =keys)
                writer.writeheader()
                writer.writerows(all_data)
            
