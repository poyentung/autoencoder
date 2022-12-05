import os
import hyperspy.api as hs
import numpy as np
import pandas as pd
import hdbscan
import pathlib
from hyperspy.signal import BaseSignal
from typing import Dict, List, Tuple, Union


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, Birch
from scipy import fftpack

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D


class PixelSegmenter(object):
    def __init__(
        self,
        latent: np.ndarray,
        dataset: ElectronDiffraction2D,
        method: str = "BayesianGaussianMixture",
        method_args: Dict = {"n_components": 8, "random_state": 4},
    ):

        self.latent = latent
        self.dataset = dataset
        self.method = method
        self.method_args = method_args
        self.sed_size = self.dataset.data.shape[:2]
        self.height = self.sed_size[0]
        self.width = self.sed_size[1]

        # Set edx and bse signal to the corresponding ones
        self.dps = self.dataset.data
        self.intensity = self.dps.reshape(self.height,self.width,-1).mean(axis=2)

        ### Get energy_axis ###
        size = self.dataset.axes_manager[2].size
        scale = self.dataset.axes_manager[2].scale
        offset = self.dataset.axes_manager[2].offset
        self.radial_axis = [((a * scale) + offset) for a in range(0, size)]

        ### Train the model ###
        if self.method == "GaussianMixture":
            self.model = GaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "BayesianGaussianMixture":
            self.model = BayesianGaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "Kmeans":
            self.model = KMeans(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "Birch":
            self.model = Birch(**method_args).partial_fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "HDBSCAN":
            self.model = hdbscan.HDBSCAN(**method_args)
            self.labels = self.model.fit_predict(latent)
            self.n_components = int(self.labels.max()) + 1 

        if self.method != "HDBSCAN":
            self.labels = self.model.predict(self.latent)

        means = []

        if len(self.dps.shape)==4:
            dataset_ravel = self.dps.reshape(-1, self.dps.shape[2], self.dps.shape[3])
        elif len(self.dps.shape)==3:
            dataset_ravel = self.dps.reshape(-1, self.dps.shape[2])

        for i in range(self.n_components):
            mean = dataset_ravel[np.where(self.labels == i)[0]].mean(axis=0)
            if len(self.dps.shape)==4:
                mean = mean.reshape(self.dps.shape[2], self.dps.shape[3])
            means.append(mean)
        self.mu = np.stack(means, axis=0)

        ### calculate cluster probability maps ###
        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            self.prob_map = self.model.predict_proba(self.latent)

        # Set color for phase visualisation
        if self.n_components <= 10:
            self._color_palette = "tab10"
            self.color_palette = "tab10"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=9)
        else:
            self._color_palette = "nipy_spectral"
            self.color_palette = "nipy_spectral"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=self.n_components - 1)

    def set_color_palette(self, cmap):
        self.color_palette = cmap

    def set_feature_list(self, new_list):
        self.peak_list = new_list
        self.sem.set_feature_list(new_list)

    @staticmethod
    def bic(
        latent,
        n_components=20,
        model="BayesianGaussianMixture",
        model_args={"random_state": 6},
    ):
        def _n_parameters(model):
            """Return the number of free parameters in the model."""
            _, n_features = model.means_.shape
            if model.covariance_type == "full":
                cov_params = model.n_components * n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "diag":
                cov_params = model.n_components * n_features
            elif model.covariance_type == "tied":
                cov_params = n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "spherical":
                cov_params = model.n_components
            mean_params = n_features * model.n_components
            return int(cov_params + mean_params + model.n_components - 1)

        bic_list = []
        for i in range(n_components):
            if model == "BayesianGaussianMixture":
                GMM = BayesianGaussianMixture(n_components=i + 1, **model_args).fit(
                    latent
                )
            elif model == "GaussianMixture":
                GMM = GaussianMixture(n_components=i + 1, **model_args).fit(latent)
            bic = -2 * GMM.score(latent) * latent.shape[0] + _n_parameters(
                GMM
            ) * np.log(latent.shape[0])
            bic_list.append(bic)
        return bic_list

    #################
    # Data Analysis #--------------------------------------------------------------
    #################

    def get_binary_map_intensity_sum(
        self,
        cluster_num:int,
        use_label:bool=False,
        threshold:float=0.8,
        denoise:bool=False,
        keep_fraction:float=0.13,
        binary_filter_threshold:float=0.2,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        if use_label == False:
            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                phase = self.model.predict_proba(self.latent)[:, cluster_num]

                if denoise == False:
                    binary_map = np.where(phase > threshold, 1, 0).reshape(
                        self.height, self.width
                    )
                    binary_map_indices = np.where(
                        phase.reshape(self.height, self.width) > threshold
                    )

                else:
                    filtered_img = np.where(phase < threshold, 0, 1).reshape(
                        self.height, self.width
                    )
                    image_fft = fftpack.fft2(filtered_img)
                    image_fft2 = image_fft.copy()

                    # Set r and c to be the number of rows and columns of the array.
                    r, c = image_fft2.shape

                    # Set to zero all rows with indices between r*keep_fraction and
                    # r*(1-keep_fraction):
                    image_fft2[
                        int(r * keep_fraction) : int(r * (1 - keep_fraction))
                    ] = 0

                    # Similarly with the columns:
                    image_fft2[
                        :, int(c * keep_fraction) : int(c * (1 - keep_fraction))
                    ] = 0

                    # Transformed the filtered image back to real space
                    image_new = fftpack.ifft2(image_fft2).real

                    binary_map = np.where(image_new < binary_filter_threshold, 0, 1)
                    binary_map_indices = np.where(image_new > binary_filter_threshold)
            else:
                binary_map = (
                    self.model.labels_
                    * np.where(self.model.labels_ == cluster_num, 1, 0)
                ).reshape(self.height, self.width)
                binary_map_indices = np.where(
                    self.model.labels_.reshape(self.height, self.width) == cluster_num
                )
        else:
            binary_map = (
                self.labels * np.where(self.labels == cluster_num, 1, 0)
            ).reshape(self.height, self.width)
            binary_map_indices = np.where(
                self.labels.reshape(self.height, self.width) == cluster_num
            )

        # Get dps in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1, 1)
        y_id = binary_map_indices[1].reshape(-1, 1)
        x_y = np.concatenate([x_id, y_id], axis=1)
        x_y_indices = tuple(map(tuple, x_y))

        total_diffraction_patterns = list()
        for x_y_index in x_y_indices:
            total_diffraction_patterns.append(self.dps[x_y_index].reshape(1, -1))

        total_diffraction_patterns = np.stack(total_diffraction_patterns, axis=0)
        intensity_sum = total_diffraction_patterns.sum(axis=0).squeeze()

        if len(self.dps.shape) == 4: # reshape back to 2D DPs
            intensity_sum = intensity_sum.reshape(self.dps.shape[2], self.dps.shape[3])

        return (binary_map, binary_map_indices, intensity_sum)

    def get_all_cluster_sum_DPs(self, normalised:bool=True) -> np.ndarray:
        cluster_sum_dps = []
        for i in range(self.n_components):
            _, _, intensity_sum = self.get_binary_map_intensity_sum(
                cluster_num=i, use_label=True
            )
            cluster_sum_dps.append(intensity_sum)
        cluster_sum_dps = np.stack(cluster_sum_dps, axis=0)

        if normalised == True:
            origin_shape = cluster_sum_dps.shape
            cluster_sum_dps = cluster_sum_dps.reshape(origin_shape[0],-1).astype('float')
            cluster_sum_dps *= cluster_sum_dps.max(axis=1, keepdims=True)**-1
            cluster_sum_dps.reshape(origin_shape)

        return cluster_sum_dps

    def get_binary_map_DPs(self, cluster_num:int, use_label:bool=True) -> Tuple[np.ndarray, np.ndarray, List]:
        binary_map, binary_map_indices, _ = self.get_binary_map_intensity_sum(cluster_num=cluster_num,use_label=use_label)
        x_id = binary_map_indices[0].reshape(-1, 1)
        y_id = binary_map_indices[1].reshape(-1, 1)
        x_y = np.concatenate([x_id, y_id], axis=1)
        x_y_indices = tuple(map(tuple, x_y))

        total_diffraction_patterns = list()
        for x_y_index in x_y_indices:
            total_diffraction_patterns.append(self.dps[x_y_index])

        return (binary_map, x_y, total_diffraction_patterns)


    def save_navigators(self,foler_path:Union[str, pathlib.Path]):
        if not os.path.isdir(foler_path):
            os.mkdir(foler_path)

        phase = self.labels.reshape(self.height, self.width)
        navigator = BaseSignal(phase)
        path_navigator=os.path.join(foler_path, 'navigator_all_cluster')
        navigator.save(path_navigator)
            
        for cluster_num in range(self.n_components):
            binary_map, _, _ = self.get_binary_map_DPs(cluster_num=cluster_num)

            navigator = BaseSignal(binary_map)
            path_navigator=os.path.join(foler_path, f'navigator_cluster_{cluster_num}')
            navigator.save(path_navigator)

    #################
    # Visualization #--------------------------------------------------------------
    #################

    def plot_latent_space(self, color=True, cmap=None):
        cmap = self.color_palette if cmap is None else cmap

        fig, axs = plt.subplots(1, 1, figsize=(4.8, 4.5), dpi=150)

        if color:
            im = axs.scatter(
                    self.latent[:, 0],
                    self.latent[:, 1],
                    c=self.labels,
                    s=5.666,
                    zorder=2,
                    alpha=0.3,
                    linewidths=0,
                    cmap=cmap,
                    norm=self.color_norm,
                )

            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                i = 0
                for pos, covar, w in zip(
                    self.model.means_, self.model.covariances_, self.model.weights_
                ):
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.14,
                        facecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        edgecolor="None",
                        zorder=-10,
                    )
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.0,
                        edgecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        facecolor="None",
                        zorder=-9,
                        lw=0.25,
                    )
                    i += 1
            
            # set colorbar
            cbar = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.04, aspect=25, 
                                format='%i', ticks=range(self.n_components)) 
            cbar.solids.set_edgecolor("face")
            cbar.outline.set_visible(False)
            cbar.solids.set(alpha=0.9)
            cbar.ax.tick_params(labelsize=6, width=0.7,length=2, which='major', direction='out')
            for t in cbar.ax.get_yticklabels():
                t.set_horizontalalignment('center')   
                t.set_x(1.2)

        else:
            axs.scatter(
                self.latent[:, 0],
                self.latent[:, 1],
                c="k",
                s=1.0,
                zorder=2,
                alpha=0.15,
                linewidths=0,
            )

        #set aspect ratio
        x_left, x_right = axs.get_xlim()
        y_low, y_high = axs.get_ylim()
        axs.set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)

        for axis in ["top", "bottom", "left", "right"]:
            axs.spines[axis].set_linewidth(1.5)
        plt.show()
        return fig

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 3):
            ax.add_patch(
                Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
            )

    def plot_cluster_distribution(self, save=None, **kwargs):
        fig, axs = plt.subplots(
            self.n_components,
            2,
            figsize=(14, self.n_components * 4.2),
            dpi=96,
            **kwargs,
        )
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        for i in range(self.n_components):
            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                prob_map_i = self.prob_map[:, i]
            else:
                prob_map_i = np.where(self.labels == i, 1, 0)

            im = axs[i, 0].imshow(
                prob_map_i.reshape(self.height, self.width), cmap="viridis"
            )

            axs[i, 0].set_title("Pixel-wise probability for cluster " + str(i))
            axs[i, 0].axis("off")

            cbar = fig.colorbar(im, ax=axs[i, 0], shrink=0.9, pad=0.025)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10, size=0)

            
            axs[i, 1].imshow(self.mu[i],cmap="viridis")
            axs[i, 1].set_title("Mean DP for cluster " + str(i))

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()

        if save is not None:
            fig.savefig(save, bbox_inches="tight", pad_inches=0.01)

    def plot_single_cluster_distribution(self, cluster_num):
        figsize = (6,3) if len(self.dps.data.shape)==4 else (8,3)
        gridspec_kw={"width_ratios": [1, 1]} if len(self.dps.data.shape)==4 else {"width_ratios": [1, 1.5]} 
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=96, gridspec_kw=gridspec_kw)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            prob_map_i = self.prob_map[:, cluster_num]
        else:
            prob_map_i = np.where(self.labels == cluster_num, 1, 0)

        im = axs[0].imshow(prob_map_i.reshape(self.height, self.width), cmap="viridis")
        axs[0].set_title("Pixel-wise probability for cluster " + str(cluster_num))
        axs[0].axis("off")

        cbar = fig.colorbar(im, ax=axs[0], shrink=0.9, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)

        if len(self.dps.data.shape)==4:
            axs[1].imshow(self.mu[cluster_num],cmap="viridis")
            axs[1].set_title("Mean DP for cluster " + str(cluster_num))
            axs[1].axis("off")
            
        elif len(self.dps.data.shape)==3:
            if self.n_components <= 10:
                axs[1].plot(
                    self.radial_axis,
                    self.mu[cluster_num],
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1)
                )
            else:
                axs[1].plot(
                    self.radial_axis,
                    self.mu[cluster_num],
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(
                        cluster_num * (self.n_components - 1) ** -1
                    )
                )
            axs[1].set_xlabel("$A^{-1}$")
            axs[1].set_ylabel("Counts")  

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        fig.set_tight_layout(True)
        plt.show()
        return fig
    
    def plot_single_cluster_profile_plotly(self, cluster_num):
#         source = pd.DataFrame(dict(x=self.radial_axis, y=self.mu[cluster_num]))
        
#         # Create a selection that chooses the nearest point & selects based on x-value
#         nearest = alt.selection(type='single', nearest=True, on='mouseover',
#                                 fields=['x'], empty='none')
        
#         # The basic line
#         line = alt.Chart(source).mark_line(interpolate='basis').encode(
#             x='x:Q',
#             y='y:Q',
#         )
        
#         # Transparent selectors across the chart. This is what tells us
#         # the x-value of the cursor
#         selectors = alt.Chart(source).mark_point().encode(
#             x='x:Q',
#             opacity=alt.value(0),
#         ).add_selection(
#             nearest
#         )
        
#         # Draw points on the line, and highlight based on selection
#         points = line.mark_point().encode(
#             opacity=alt.condition(nearest, alt.value(1), alt.value(0))
#         )
            
#         # Draw text labels near the points, and highlight based on selection
#         text = line.mark_text(align='left', dx=5, dy=-5).encode(
#             text=alt.condition(nearest, 'x:Q', alt.value(' '))
#         )
        
#         # Draw a rule at the location of the selection
#         rules = alt.Chart(source).mark_rule(color='gray').encode(
#             x='x:Q',
#         ).transform_filter(
#             nearest
#         )

#         # Put the five layers into a chart and bind the data
#         return alt.layer(
#                     line, selectors, points, rules, text
#                 ).properties(
#                     width=600, height=300
#                 )
        
        fig = go.Figure(
                data=go.Scatter(x=self.radial_axis, y=self.mu[cluster_num],mode='lines'),
                layout=go.Layout(
                        title="Mean profile",
                        title_x=0.5,
                        xaxis_title="$A^{-1}$",
                        yaxis_title="Counts",
                        width=700,
                        height=400,
                    ),
                )
        fig.update_layout(showlegend=False)
        fig.update_layout(template="simple_white")
        fig.show()

    def plot_phase_map(self, cmap=None):
        cmap = self.color_palette if cmap is None else cmap
        img = self.intensity
        phase = self.model.predict(self.latent).reshape(self.height, self.width)

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4), dpi=100)

        # for i in not_to_show:
        #     phase[np.where(phase==i)]=0

        axs[0].imshow(img, cmap="gray", interpolation="none")
        axs[0].set_title("Sum intensity")
        axs[0].axis("off")

        axs[1].imshow(img, cmap="gray", interpolation="none", alpha=1.0)

        if self.n_components <= 10:
            axs[1].imshow(
                phase,
                cmap=self.color_palette,
                interpolation="none",
                norm=self.color_norm,
                alpha=0.75,
            )
        else:
            axs[1].imshow(
                phase,
                cmap=self.color_palette,
                interpolation="none",
                alpha=0.6,
                norm=self.color_norm,
            )
        axs[1].axis("off")
        axs[1].set_title("Cluster map")

        fig.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.show()
        return fig

    def plot_binary_map_cluster_DP(
        self, cluster_num, **kwargs
    ):

        binary_map, binary_map_indices, intensity_sum = self.get_binary_map_intensity_sum(
            cluster_num, use_label=True
        )
        
        gridspec_kw={"width_ratios": [1, 1, 1]} if len(self.dps.data.shape)==4 else {"width_ratios": [1, 1, 1.7]} 
        figsize = (9, 3) if len(self.dps.data.shape)==4 else (10, 3)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=figsize,
            dpi=96,
            gridspec_kw=gridspec_kw,
            **kwargs,
        )

        phase_color = plt.cm.get_cmap(self.color_palette)(
            cluster_num * float(self.n_components - 1) ** -1
        )
        c = mcolors.ColorConverter().to_rgb

        if (self.n_components > 10) and (cluster_num == 0):
            cmap = make_colormap([c("k"), c("w"), 1, c("w")])
        else:
            cmap = make_colormap([c("k"), phase_color[:3], 1, phase_color[:3]])

        axs[0].imshow(binary_map, cmap=cmap)
        axs[0].set_title(f"Binary map (cluster {cluster_num})", fontsize=10)
        axs[0].axis("off")
        axs[0].set_aspect("equal", "box")

        background = self.intensity
        axs[1].imshow(background, cmap="gray", interpolation="none", alpha=1)
        axs[1].scatter(
            binary_map_indices[1], binary_map_indices[0], c="r", alpha=0.5, s=1.5
        )
        axs[1].grid(False)
        axs[1].axis("off")
        axs[1].set_title("Intensity + Binary Map", fontsize=10)

        if len(self.dps.data.shape)==4:
            axs[2].imshow(intensity_sum, cmap='viridis')
            axs[2].axis("off")
            axs[2].set_title("Sum DP", fontsize=10)

        elif len(self.dps.data.shape)==3:
            if self.n_components <= 10:
                axs[2].plot(
                    self.radial_axis,
                    intensity_sum,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
                )
            else:
                axs[2].plot(
                    self.radial_axis,
                    intensity_sum,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(
                        cluster_num * (self.n_components - 1) ** -1
                    ),
                )
            axs[2].set_xlabel("$A^{-1}$")
            axs[2].set_ylabel("Counts")  

        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()
        plt.show()
        return fig

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)