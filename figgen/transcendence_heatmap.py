# %%
import os
from glicko2 import GlickoCalc
from figgen import DataAnalyzer

os.environ["DISPLAY"] = ""

import pandas as pd

class TranscendenceHeatmapAnalyzer(DataAnalyzer):
    def fetch_and_process_skill_level_data_for_temperature(self, df: pd.DataFrame):
        row = df.iloc[0]

        glicko = GlickoCalc()

        stockfish_index = "one" if row["player_one"].startswith("Stockfish") else "two"
        stockfish_level = int(row[f"player_{stockfish_index}"].split(" ")[1])

        for idx, row in df.iterrows():
            if row["player_one"].startswith("Stockfish"):
                nanogpt_index = "two"
                stockfish_index = "one"
            else:
                nanogpt_index = "one"
                stockfish_index = "two"

            if row[f"player_{nanogpt_index}_score"] == "1":
                glicko.glicko2_update(0, stockfish_level)
            elif row[f"player_{nanogpt_index}_score"] == "0":
                glicko.glicko2_update(1, stockfish_level)
            else:
                glicko.glicko2_update(2, stockfish_level)

        return glicko.current_elo, glicko.current_deviation, stockfish_level  # TODO        


    def calc(self, table, formatted_data, sf_level, my_elo, dev, groupby, y_label):
        # level_pos = table.loc[0, "game_title"].find("Stockfish ")
        # sf_name = table.loc[0,"game_title"][level_pos : level_pos + 12]
        # sf_level = sf_name.split(" ")[1]   
        nanogpt_elo = table.loc[0, "nanogpt_elo"]
        elo_level = my_elo
        stockfish_elo = table.loc[0, "stockfish_elo"]
        
        formatted_data += [
            {
                "nanogpt_elo": nanogpt_elo,
                f"{groupby}": elo_level,
                "deviation": dev,
                f"{y_label}": stockfish_elo,
            },
            # {
            #     "nanogpt_elo": nanogpt_elo,
            #     f"{groupby}": 2 * elo_level - dev,
            #     f"{groupby}": dev,
            #     f"{y_label}": stockfish_elo,
            # },  # super hacky
        ]
        
        # formatted_data.append({
        #     "nanogpt_elo": nanogpt_elo,
        #     # f"{groupby}": sf_level,
        #     f"{groupby}": elo_level,
        #     f"{y_label}": stockfish_elo, 
        # })

    def get_tables(self, run_id):
        runs = self.get_runs([run_id])
        artifacts = runs[0].logged_artifacts()
        table_list = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet":
                break
            print("Table Name: ", table_name)
            if "#fx+" in table_name:
                continue
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
            table_list.append(df)
            
        return table_list


# %%
import pandas as pd

if __name__ == "__main__":
    runs_ids_770_2000_eval = ["urkdh2hg", "oez2rf1k", "izxet7yw", "it9rgbli", "im5ixhih", "bpp94f7q", "3oip3139", "1qrfyvmv"]
    
    analyzer = TranscendenceHeatmapAnalyzer(
        wandb_entity="project-eval", wandb_project="full-eval-770_2000-Eval-Full"
    )

    all_tables = []
    groupby = "Stockfish_Skill_Level" 
    x_label = "nanogpt_elo"
    y_label = "stockfish_elo"
    for run_id in runs_ids_770_2000_eval:
        all_tables += analyzer.get_tables(run_id)
    
    formatted_data = []
    for tab in all_tables:
        (
            nanogpt_elo,
            dev,
            sf_level,
        ) = analyzer.fetch_and_process_skill_level_data_for_temperature(tab)
        analyzer.calc(tab, formatted_data, sf_level, nanogpt_elo, dev, groupby, y_label)
        
    def temperature_sampling_experiment(groupby, y_label, data):
        analyzer.visualize_heatmap_groupby(
            f"Glicko Calc NanoGPT Elo Across Elo Matchups",
            x_label,
            y_label,
            groupby,
            data,
            x_label="NanoGPT Elo",
            y_label="Stockfish Elo",
        )

    temperature_sampling_experiment(groupby, y_label, formatted_data)