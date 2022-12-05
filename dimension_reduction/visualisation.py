import random
import numpy as np
import altair as alt
import pandas as pd
import matplotlib.colors as mcolors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from .clustering import PixelSegmenter

import os
from typing import List, Dict
import hyperspy.api as hs
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display


def _plot_latent(dataset: ElectronDiffraction2D, latent:np.ndarray, ratio_to_be_shown:float=1.0):

    dp_size = dataset.data.shape[:2]
    x_id, y_id = np.meshgrid(range(dp_size[0]), range(dp_size[1]))
    x_id = x_id.ravel().reshape(-1, 1)
    y_id = y_id.ravel().reshape(-1, 1)
    z_id = dataset.data.reshape(dp_size[0],dp_size[1],-1).sum(axis=2).reshape(-1, 1)
    z_id = z_id / z_id.max()

    combined = np.concatenate(
        [
            x_id,
            y_id,
            z_id,
            latent,
            # dataset.data.reshape(-1, dataset.data.shape[-1]).round(2)
        ],
        axis=1,
    )

    if ratio_to_be_shown==1:
        sampled_combined = combined
    else:
        sampled_combined = random.choices(
            combined, k=int(latent.shape[0] // (ratio_to_be_shown ** -1))
        )
        sampled_combined = np.array(sampled_combined)

    source = pd.DataFrame(
        sampled_combined,
        columns=["x_id", "y_id", "z_id", "x", "y"],
        index=pd.RangeIndex(0, sampled_combined.shape[0], name="pixel"),
    )   

    # Plotting using Altair
    alt.data_transformers.disable_max_rows()

    # Brush
    brush = alt.selection(type="interval")

    interaction = alt.selection(
        type="interval",
        bind="scales",
        on="[mousedown[event.shiftKey], mouseup] > mousemove",
        translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
        zoom="wheel![event.shiftKey]",
    )

    # Points
    points = (
        alt.Chart(source)
        .mark_circle(size=4.0,color='red')
        .encode(
            x="x:Q",
            y="y:Q",  # use min extent to stabilize axis title placement
            # color=alt.Color( scale=alt.Scale(scheme="black")),
            # color=alt.Color(
            #     "Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)
            # ),
            color=alt.condition(brush, alt.value('red'), alt.value('grey')),
            opacity=alt.condition(brush, alt.value(0.5), alt.value(0.5)),
            tooltip=[
                alt.Tooltip("x:Q", format=",.2f"),
                alt.Tooltip("y:Q", format=",.2f"),
            ],
        )
        .properties(width=450, height=450)
        .properties(title=alt.TitleParams(text="Latent space"))
        .add_selection(
            brush,
            interaction
            )
    )

    # Heatmap
    intensity_df = pd.DataFrame(
        {"x_intensity": x_id.ravel(), "y_intensity": y_id.ravel(), "z_intensity": z_id.ravel()}
    )
    intensity = (
        alt.Chart(intensity_df)
        .mark_circle(size=6)
        .encode(
            x=alt.X("x_intensity:O", axis=None),
            y=alt.Y("y_intensity:O", axis=None),
            color=alt.Color(
                "z_intensity:Q", scale=alt.Scale(scheme="greys", domain=[1.0, 0.0])
            ),
        )
        .properties(width=dp_size[0]*2, height=dp_size[1]*2)
    )
    heatmap = (
        alt.Chart(source)
        .mark_circle(size=5.0,color='red')
        .encode(
            x=alt.X("x_id:O", axis=None),
            y=alt.Y("y_id:O", axis=None),
            # color=alt.Color(
            #     "z_bse:Q", scale=alt.Scale(scheme="Blues", domain=[0.0, 1.0])
            # ),
            opacity=alt.condition(brush, alt.value(0.6), alt.value(0)),
        )
        .properties(width=dp_size[0]*2, height=dp_size[1]*2)
        .add_selection(brush)
    )

    heatmap_intensity = intensity + heatmap

    final_widgets = [points, heatmap_intensity]

    # Build chart
    chart = (
        alt.hconcat(*final_widgets)
        .resolve_legend(color="independent")
        .configure_view(strokeWidth=0)
    )

    return chart

def view_bic(
    latent: np.ndarray,
    n_components: int = 20,
    model: str = "BayesianGaussianMixture",
    model_args: Dict = {"random_state": 6},
):
    bic_list = PixelSegmenter.bic(latent, n_components, model, model_args)
    fig = go.Figure(
        data=go.Scatter(
            x=np.arange(1, n_components + 1, dtype=int),
            y=bic_list,
            mode="lines+markers",
        ),
        layout=go.Layout(
            title="",
            title_x=0.5,
            xaxis_title="Number of component",
            yaxis_title="BIC",
            width=800,
            height=600,
        ),
    )

    fig.update_layout(showlegend=False)
    fig.update_layout(template="simple_white")
    fig.update_traces(marker_size=12)
    fig.show()
    save_csv(pd.DataFrame(data={"bic": bic_list}))

def save_csv(df):
    text = widgets.Text(
        value="file_name.csv",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
    )

    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            df.to_csv(text.value)
            print("save the csv to", text.value)

    button.on_click(save_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

def save_fig(fig):
    file_name = widgets.Text(
        value="figure_name.tif",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    folder_name = widgets.Text(
        value="results",
        placeholder="Type something",
        description="Folder name:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    dpi = widgets.BoundedIntText(
        value="96",
        min=0,
        max=300,
        step=1,
        description="Set dpi:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    pad = widgets.BoundedFloatText(
        value="0.01",
        min=0.0,
        description="Set pad:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            if not os.path.isdir(folder_name.value):
                os.mkdir(folder_name.value)
            if isinstance(fig, mpl.figure.Figure):
                save_path = os.path.join(folder_name.value, file_name.value)
                fig.savefig(
                    save_path, dpi=dpi.value, bbox_inches="tight", pad_inches=pad.value
                )
                print("save figure to", file_name.value)
            else:
                initial_file_name = file_name.value.split(".")
                folder_for_fig = os.path.join(folder_name.value, initial_file_name[0])
                if not os.path.isdir(folder_for_fig):
                    os.mkdir(folder_for_fig)
                for i, single_fig in enumerate(fig):
                    save_path = os.path.join(
                        folder_for_fig,
                        f"{initial_file_name[0]}_{i:02}.{initial_file_name[1]}",
                    )
                    single_fig.savefig(
                        save_path,
                        dpi=dpi.value,
                        bbox_inches="tight",
                        pad_inches=pad.value,
                    )
                print("save all figure to folder:", folder_for_fig)

    button.on_click(save_to)
    all_widgets = widgets.HBox(
        [folder_name, file_name, dpi, pad, button], layout=Layout(width="auto")
    )
    display(all_widgets)
    display(out)
