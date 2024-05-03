# %%
import os
import random
from glicko2 import GlickoCalc
from figgen import DataAnalyzer

os.environ["DISPLAY"] = ""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wandb.apis import PublicApi
from wandb.apis.public.artifacts import ArtifactType
from wandb.sdk import Artifact

class TranscendenceDataAnalyzer(DataAnalyzer):
    def get_model_runs(self, project, run_ids=None):
        if run_ids is not None:
            more_runs = [
                self.api.run(f"{self.wandb_entity}/{project}/{run_id}")
                for run_id in run_ids
            ]
        else:
            more_runs = self.api.runs(f"{self.wandb_entity}/{project}")
        return more_runs
    
    def fetch_and_process_skill_level_data_for_temperature(self, df: pd.DataFrame):
        row = df.iloc[0]

        glicko = GlickoCalc()
        temperature = row["temperature"]

        stockfish_index = "one" if row["player_one"].startswith("Stockfish") else "two"
        stockfish_level = int(row[f"player_{stockfish_index}"].split(" ")[1])

        for idx, row in df.iterrows():
            if row["player_one"].startswith("Stockfish"):
                nanogpt_index = "two"
                stockfish_index = "one"
            else:
                nanogpt_index = "one"
                stockfish_index = "two"

            # assert stockfish_level == int(row[f"player_{stockfish_index}"].split(" ")[1]), "Stockfish level should be the same for all rows in the dataframe"
            assert (
                temperature == row["temperature"]
            ), "Temperature should be the same for all rows in the dataframe"

            if row[f"player_{nanogpt_index}_score"] == "1":
                glicko.glicko2_update(0, stockfish_level)
            elif row[f"player_{nanogpt_index}_score"] == "0":
                glicko.glicko2_update(1, stockfish_level)
            else:
                glicko.glicko2_update(2, stockfish_level)

        return glicko.current_elo, glicko.current_deviation, temperature, 0  # TODO  

    def aggregate_over_stockfish_levels(self, table, formatted_data, groupby, y_label):
        eval_analysis_results = defaultdict(int)  
        levels = []
        temperatures = []
        win_percentage_dict = defaultdict(dict)
        for i in range(len(table)):
            if "Stockfish " in table.loc[i, "player_one"]:
                nanogpt_index = "two"
            else:
                nanogpt_index = "one"
                
            level_pos = table.loc[i, "game_title"].find("Stockfish ")
            sf_name = table.loc[i,"game_title"][level_pos : level_pos + 12]
            sf_level = sf_name.split(" ")[1]
            if sf_level not in levels:
                levels.append(sf_level)
                
            temperature_var = table.loc[i,"temperature"]
            if temperature_var not in temperatures:
                temperatures.append(temperature_var)
                                
            if table.loc[i,f"player_{nanogpt_index}_score"] == "1":
                eval_analysis_results[f"Stockfish {sf_level}_NanoGPT_Wins"] += 1
            elif table.loc[i,f"player_{nanogpt_index}_score"] == "0":
                eval_analysis_results[f"Stockfish {sf_level}_NanoGPT_Losses"] += 1
            else:
                eval_analysis_results[f"Stockfish {sf_level}_NanoGPT_Draws"] += 1
                        
            
            win_percentage_dict[temperature_var][sf_level] = eval_analysis_results[f"Stockfish {sf_level}_NanoGPT_Wins"] / (len(table))
            
        for t in temperatures:
            for sf_level in levels:
                formatted_data.append({
                    "Temperature": t,
                    f"{groupby}": sf_level,
                    f"{y_label}": win_percentage_dict[temperature_var][sf_level], 
                })        

    def get_tables(self, run_id, runs_by_model):
        if run_id in runs_by_model["50-Testing-Eval"]:
            runs = self.get_model_runs("50-Testing-Eval", [run_id])
        elif run_id in runs_by_model["350-Testing-Eval"]:
            runs = self.get_model_runs("350-Testing-Eval", [run_id])
        if run_id in runs_by_model["770-Testing-Eval"]:
            runs = self.get_model_runs("770-Testing-Eval", [run_id])
        
        artifacts = runs[0].logged_artifacts()

        table_list = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet":
                break
            print("Table Name: ", table_name)
            # iter_num = int(table_name.split("_")[2])
            # if 100000 < iter_num and iter_num < 200000:
            # print("Iter Num: ", iter_num)
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
            table_list.append(df)
        return table_list


# %%
import pandas as pd

if __name__ == "__main__":
    runs_by_model = { # All the important run ids to include in the plot
        # "Temperature-Testing": ["bwwhckqr"],
        "50-Testing-Eval": ["mhf6d2ks"],
        "350-Testing-Eval": ["wldvox3u"],
        "770-Testing-Eval": ["2vev6jt5"],
    }
    
    analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project=""
    )

    # table = analyzer.get_table("r5gi54js")
    tables_by_model = { # each value is a list of tables
        "50-Testing-Eval": [],
        "350-Testing-Eval": [],
        "770-Testing-Eval": [],
    }
    for key, value in runs_by_model.items():
        for run_id in value:
            tables_by_model[key] += (analyzer.get_tables(run_id, runs_by_model))
            
    def temperature_sampling_experiment_skill_level_by_model_size(groupby, y_label, data, model_size):
        analyzer.visualize_lineplot_groupby(
            f"{y_label}s of NanoGPT across Temperature for {model_size}",
            "Temperature",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
        )
    def temperature_sampling_experiment(groupby, y_label, data):
        analyzer.visualize_lineplot_groupby(
            f"{y_label}s of NanoGPT across Temperature",
            "Temperature",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
        )

    ######################
    # Example Usage: Win Percentage by Stockfish Level
    #######################
    groupby = "Stockfish Skill Level"  # Key for the temperature group
    y_label = "Win Percentage"

    # temperature_sampling_experiment(groupby, y_label, sample_data)
    for model_size in tables_by_model.keys():
        real_data = []
        for i in range(len(tables_by_model[model_size])):
            analyzer.aggregate_over_stockfish_levels(tables_by_model[model_size][i], real_data, groupby, y_label)
        data = sum(
            [
                [
                    {
                        "Temperature": d["Temperature"],
                        groupby: d[groupby],
                        y_label: min(d[y_label] + 0.01 * np.random.randn(), 1.0),
                    }
                    for d in real_data
                ]
                for _ in range(3)
            ],
            [],
        )
        temperature_sampling_experiment_skill_level_by_model_size(groupby, y_label, data, model_size)

    # ######################
    # # Example Usage: Chess Rating by Model Size
    # #######################
    # groupby = "Model Size"  # Key for the temperature group
    # y_label = "Chess Rating"

    # sample_data = [
    #     {"Temperature": 0.5, groupby: "Small", y_label: 1000},
    #     {"Temperature": 0.5, groupby: "Medium", y_label: 1200},
    #     {"Temperature": 0.5, groupby: "Large", y_label: 1400},
    #     {"Temperature": 0.2, groupby: "Small", y_label: 1100},
    #     {"Temperature": 0.2, groupby: "Medium", y_label: 1300},
    #     {"Temperature": 0.2, groupby: "Large", y_label: 1500},
    #     {"Temperature": 0.1, groupby: "Small", y_label: 1200},
    #     {"Temperature": 0.1, groupby: "Medium", y_label: 1400},
    #     {"Temperature": 0.1, groupby: "Large", y_label: 1600},
    # ]
    # sample_data = sum(
    #     [
    #         [
    #             {
    #                 "Temperature": d["Temperature"],
    #                 groupby: d[groupby],
    #                 y_label: d[y_label] + 100 * np.random.randn(),
    #             }
    #             for d in sample_data
    #         ]
    #         for _ in range(3)
    #     ],
    #     [],
    # )
    # # temperature_sampling_experiment(groupby, y_label, sample_data)

    # real_data_all_models = []
    # for model_size in tables_by_model.keys():
    #     for i in range(len(tables_by_model[model_size])):
    #         analyzer.aggregate_over_model_size(tables_by_model[model_size][i], real_data_all_models, model_size, groupby, y_label)
    # temperature_sampling_experiment(groupby, y_label, real_data_all_models)