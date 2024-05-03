import os
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv


class DataAnalyzer:
    def __init__(
        self,
        wandb_entity,
        wandb_project,
        min_length=100,
        color_scheme="viridis",
        export_to_wandb=False,
    ):
        load_dotenv()
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.api_key = os.environ["WANDB_API_KEY"]
        self.api = wandb.Api()
        self.histories = {}
        self.min_length = min_length
        self.color_scheme = color_scheme
        self.export_to_wandb = export_to_wandb
        if self.export_to_wandb and self.api_key:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def get_runs(self, run_ids=None):
        if run_ids is not None:
            self.runs = [
                self.api.run(f"{self.wandb_entity}/{self.wandb_project}/{run_id}")
                for run_id in run_ids
            ]
        else:
            self.runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")
        return self.runs

    def send_to_wandb(self, fig, title):
        if self.export_to_wandb:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_path = tmpfile.name
                fig.savefig(fig_path)
                wandb.log({title: wandb.Image(fig_path)})
                os.unlink(fig_path)

    def get_histories(self):
        for run in self.runs:
            self.histories[run.id] = run.history()

    def visualize_lineplot_groupby(
        self,
        title: str,
        x_key: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
    ):
        """Visualize a dataframe with this form. Here, Temperature is the x-axis, Rating is the y-axis, and Stockfish Skill Level is the group key.:

        stockfish_groupby = "Stockfish Skill Level" # Key for the temperature group
        sample_data = [
            {"Temperature": 0.3, stockfish_groupby: "0", "Rating": 1000},
            {"Temperature": 0.3, stockfish_groupby: "1", "Rating": 1200},
            {"Temperature": 0.3, stockfish_groupby: "2", "Rating": 1400},
            {"Temperature": 0.2, stockfish_groupby: "0", "Rating": 1100},
            {"Temperature": 0.2, stockfish_groupby: "1", "Rating": 1300},
            {"Temperature": 0.2, stockfish_groupby: "2", "Rating": 1500},
            {"Temperature": 0.1, stockfish_groupby: "0", "Rating": 1200},
            {"Temperature": 0.1, stockfish_groupby: "1", "Rating": 1400},
            {"Temperature": 0.1, stockfish_groupby: "2", "Rating": 1600},
        ]

        # add noise to data for error bars
        data = pd.DataFrame.from_dict(
            sum(
                [
                    [
                        {
                            "Temperature": d["Temperature"],
                            stockfish_groupby: d[stockfish_groupby],
                            "Rating": d["Rating"] + 100 * np.random.randn(),
                        }
                        for d in sample_data
                    ]
                    for _ in range(3)
                ],
                [],
            )
        )

        visualize_lineplot_groupby("Chess Ratings of NanoGPT across Temperature", "Temperature", "Rating", stockfish_groupby, data, x_ticks_by_data=True)

        """
        groups = data[group_key].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
            )

            plt.title(title, fontsize="large")
            plt.xlabel(x_label or x_key.capitalize(), fontsize="large")
            plt.ylabel(y_label or y_key.capitalize(), fontsize="large")

            # plt.xlim(left=0, right=self.min_length)
            if x_ticks_by_data:
                plt.xticks(data[x_key].unique())

            plt.ylim(bottom=0)
            plt.grid(True)
            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")
                
    def visualize_heatmap_groupby(
        self,
        title: str,
        x_key: str,
        y_key: str,
        groupby: str,
        data: list,
        x_label: str = None,
        y_label: str = None,
    ):
        
        fig, ax = plt.subplots(figsize=(20, 20))

        # Create DataFrame and process it
        df_original = pd.DataFrame(data)
        df = df_original.copy()

        # Convert column to numeric and drop NaNs
        df['Stockfish_Skill_Level'] = pd.to_numeric(df['Stockfish_Skill_Level'], errors='coerce')
        df.dropna(inplace=True)

        # Group by necessary columns
        # print(df)
        # df = df[f'{x_key}', f'{y_key}'].unique()
        df = df.groupby([f'{x_key}', f'{y_key}']).mean().reset_index()
        annotations = df.copy()
        print("\u00B1")
        for i in range(len(annotations)):
            # annotations.loc[i, annotations[groupby]] = f'{str(round(annotations.loc[i, annotations[groupby]]))} \u00B1 {str(round(annotations.loc[i, annotations['deviation']]))}'
            annotations.loc[i, groupby] = f"{round(annotations.loc[i, groupby])} \u00B1 {round(annotations.loc[i, 'deviation'])}"

        print(df)
        print(annotations)

        # Pivot into matrix form
        pivot_table = df.pivot(index=y_key, columns=x_key, values=groupby)
        pivot_table_annotations = annotations.pivot(index=y_key, columns=x_key, values=groupby)
        print(pivot_table)
        print(pivot_table_annotations)
        assert pivot_table.shape == pivot_table_annotations.shape
            
        sns.set_theme(font_scale=1.8)
        # sns.heatmap(pivot_table, annot=True, fmt=".0f", annot_kws={'size': 30}, cmap="Blues", vmin= 1500, vmax=1800, square=True, linewidth=0.3, cbar_kws={"shrink": .8})
        sns.heatmap(pivot_table, annot=pivot_table_annotations, fmt='', annot_kws={'size': 30}, cmap="Blues", vmin= 1500, vmax=1800, square=True, linewidth=0.3, cbar_kws={"shrink": .8})
        
        
        plt.xlabel(x_label, fontsize=35, labelpad=40)
        plt.xticks(fontsize=25)
        plt.ylabel(y_label, fontsize=35, labelpad=40)
        plt.yticks(fontsize=25)
        plt.title(title, fontsize=40, pad=40)
        # plt.show()
        
        if self.export_to_wandb:
            self.send_to_wandb(fig, title)
        else:
            plt.savefig(f"{title}.png")
