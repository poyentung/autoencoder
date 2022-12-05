from . import visualisation as visual
from .clustering import PixelSegmenter

import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict
import hyperspy.api as hs
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D

# to make sure the plot function works
from plotly.offline import init_notebook_mode

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
    
def view_latent_space(ps: PixelSegmenter, color=True):
    colors = []
    cmap = plt.get_cmap(ps.color_palette)
    for i in range(ps.n_components):
        colors.append(mpl.colors.to_hex(cmap(i * (ps.n_components - 1) ** -1)[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for i, c in enumerate(colors):
        color_pickers.append(
            widgets.ColorPicker(
                value=c, description=f"cluster_{i}", layout=layout_format
            )
        )

    newcmp = mpl.colors.ListedColormap(colors, name="new_cmap")
    out = widgets.Output()
    with out:
        fig = ps.plot_latent_space(color=color, cmap=None)
        plt.show()
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            newcmp = mpl.colors.ListedColormap(color_for_map, name="new_cmap")
            ps.set_color_palette(newcmp)
            fig = ps.plot_latent_space(color=color, cmap=newcmp)
            save_fig(fig)

    button = widgets.Button(
        description="Set", layout=Layout(flex="8 1 0%", width="auto")
    )
    button.on_click(change_color)

    # Reset button
    def reset(_):
        out.clear_output()
        with out:
            fig = ps.plot_latent_space(color=color, cmap=ps._color_palette)
            save_fig(fig)

    button2 = widgets.Button(
        description="Reset", layout=Layout(flex="2 1 0%", width="auto")
    )
    button2.on_click(reset)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    button_box = widgets.HBox([button, button2])
    color_box = widgets.VBox(
        [widgets.VBox(color_list), button_box],
        layout=Layout(flex="2 1 0%", width="auto"),
    )
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))
    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def ckeck_latent_space(ps: PixelSegmenter, color:bool=True, ratio_to_be_shown:float=1.0):

    dataset = ps.dataset
    latent = ps.latent
    dp_size = dataset.data.shape[:2]
    x_id, y_id = np.meshgrid(range(dp_size[0]), range(dp_size[1]))
    x_id = x_id.ravel().reshape(-1, 1)
    y_id = y_id.ravel().reshape(-1, 1)
    z_id = dataset.data.reshape(dp_size[0],dp_size[1],-1).sum(axis=2).reshape(-1, 1)
    z_id = z_id / z_id.max()


    # create color codes
    phase_colors = []
    for i in range(ps.n_components):
        r, g, b = cm.get_cmap(ps.color_palette)(i * (ps.n_components - 1) ** -1)[:3]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        phase_colors.append(color)

    if ps.method != 'HDBSCAN':
        domain = [i for i in range(ps.n_components)]
        range_ = phase_colors
    else:
        r, g, b = colors.to_rgba('silver')[:3]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        domain = [i for i in range(-1, ps.n_components)]
        range_ = [color] + phase_colors

    combined = np.concatenate(
        [
            x_id,
            y_id,
            z_id,
            latent,
            ps.labels.reshape(-1, 1),
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
        columns=["x_id", "y_id", "z_id", "x", "y", "Cluster_id"],
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
    c = alt.Color("Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)) if color else alt.condition(brush, alt.value('red'), alt.value('grey'))
    points = (
        alt.Chart(source)
        .mark_circle(size=6.0,color='red')
        .encode(
            x="x:Q",
            y="y:Q",  
            color=c,
            opacity=alt.condition(brush, alt.value(0.5), alt.value(0.5)),
            tooltip=[
                "Cluster_id:N",
                alt.Tooltip("x:Q", format=",.2f"),
                alt.Tooltip("y:Q", format=",.2f"),
            ],
        )
        .properties(width=600, height=600)
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
        .mark_square(size=9)
        .encode(
            x=alt.X("x_intensity:O", axis=None),
            y=alt.Y("y_intensity:O", axis=None),
            color=alt.Color(
                "z_intensity:Q", scale=alt.Scale(scheme="greys", domain=[1.0, 0.0])
            ),
        )
        .properties(width=dp_size[0]*3, height=dp_size[1]*3)
    )
    heatmap = (
        alt.Chart(source)
        .mark_square(size=8,color='red')
        .encode(
            x=alt.X("x_id:O", axis=None),
            y=alt.Y("y_id:O", axis=None),
            color=c,
            opacity=alt.condition(brush, alt.value(0.75), alt.value(0)),
        )
        .properties(width=dp_size[0]*3, height=dp_size[1]*3)
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


def show_cluster_distribution(ps: PixelSegmenter, **kwargs):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select_cluster = widgets.SelectMultiple(options=["All"] + cluster_options)
    plots_output = widgets.Output()
    
    # if len(ps.dps.data.shape)==3:
    #     plot_output_detail = widgets.Output()

    all_fig = []
    # with plots_output:
    #     for i in range(ps.n_components):
    #             fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
    #             all_fig.append(fig)

    def eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if change.new == ("All",):
                for i in range(ps.n_components):
                    fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
            else:
                for cluster in change.new:
                    fig = ps.plot_single_cluster_distribution(
                        cluster_num=int(cluster.split("_")[1]), **kwargs
                    )
        # if len(ps.dps.data.shape)==3:
        #     plot_output_detail.clear_output()
        #     with plot_output_detail:
        #         if change.new != ("All",):
        #             for cluster in change.new:
        #                 ps.plot_single_cluster_profile_plotly(cluster_num=int(cluster.split("_")[1]))

    multi_select_cluster.observe(eventhandler, names="value")
    
    display(multi_select_cluster)
    save_fig(all_fig)
    # if len(ps.dps.data.shape)==3:
    #     tab_list = [plots_output, plot_output_detail]
    #     tab = widgets.Tab(tab_list)
    #     tab.set_title(0, "location")
    #     tab.set_title(1, "detailed profile")
    #     display(tab)
    display(plots_output)

def view_cluster_binary(ps: PixelSegmenter):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()

    figs = []
    with plots_output:
        for cluster in cluster_options:
            fig = ps.plot_binary_map_cluster_DP(
                cluster_num=int(cluster.split("_")[1])
            )
            figs.append(fig)

    def eventhandler(change):
        plots_output.clear_output()

        with plots_output:
            for cluster in change.new:
                fig = ps.plot_binary_map_cluster_DP(
                    cluster_num=int(cluster.split("_")[1])
                )

    multi_select.observe(eventhandler, names="value")

    display(multi_select)
    save_fig(figs)
    display(plots_output)

def view_phase_map(ps):
    colors = []
    cmap = plt.get_cmap(ps.color_palette)
    for i in range(ps.n_components):
        colors.append(mpl.colors.to_hex(cmap(i * (ps.n_components - 1) ** -1)[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for i, c in enumerate(colors):
        color_pickers.append(
            widgets.ColorPicker(
                value=c, description=f"cluster_{i}", layout=layout_format
            )
        )

    newcmp = mpl.colors.ListedColormap(colors, name="new_cmap")
    out = widgets.Output()
    with out:
        fig = ps.plot_phase_map(cmap=None)
        plt.show()
        save_fig(fig)