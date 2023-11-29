from collections.abc import Iterable
from mpl_toolkits.mplot3d.art3d import Line3D
from typing import Optional, Tuple, Union
import abc
import cebra
import h5py
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import subprocess
import torch

SOURCE = "/home/alicia/data3_personal/cebra_outputs"
DESTINATION = '/Users/kunyangalicialu/Documents/research/cebra'

class _BasePlot:
    """Base plotting class.

    Attributes:
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
    """

    def __init__(self, axis: Optional[matplotlib.axes.Axes], figsize: tuple,
                 dpi: int):
        if axis is None:
            self.fig = plt.figure(figsize=figsize, dpi=dpi)

    @abc.abstractmethod
    def _define_ax(
            self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        raise NotImplementedError()

    @abc.abstractmethod
    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        raise NotImplementedError()


class _EmbeddingPlot(_BasePlot):
    """Plot a CEBRA embedding in a 3D or 2D dimensional space.

    Attributes:
        embedding: A matrix containing the feature representation computed with CEBRA.
        embedding_labels: The labels used to map the data to color. It can be a vector that is the
            same sample size as the embedding, associating a value to each of the sample, either discrete
            or continuous or string, either `time`, then the labels while color the embedding based on
            temporality, or a string that can be interpreted as a RGB(A) color, then the embedding will be
            uniformly display with that unique color.
        ax: Optional axis to create the plot on.
        idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
            embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
            dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
            (e.g., (2, 4, 5)).
        markersize: The marker size.
        alpha: The marker blending, between 0 (transparent) and 1 (opaque).
        cmap: The Colormap instance or registered colormap name used to map scalar data to colors.
            It will be ignored if `embedding_labels` is set to a valid RGB(A).
        title: The title on top of the embedding.
        axis: Optional axis to create the plot on.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.

    """

    def __init__(
        self,
        embedding: Union[npt.NDArray, torch.Tensor],
        embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]],
        idx_order: Optional[Tuple[int]],
        markersize: float,
        alpha: float,
        cmap: str,
        title: str,
        axis: Optional[matplotlib.axes.Axes],
        figsize: tuple,
        dpi: int,
    ):
        super().__init__(axis, figsize, dpi)
        self._define_plot_dim(embedding, idx_order)
        self._define_ax(axis)
        self.embedding = embedding
        self.embedding_labels = embedding_labels
        self.idx_order = self._define_idx_order(idx_order)
        self.markersize = markersize
        self.alpha = alpha
        self.cmap = cmap
        self.title = title

    def _define_plot_dim(
        self,
        embedding: Union[npt.NDArray, torch.Tensor],
        idx_order: Optional[Tuple[int]],
    ):
        """Define the dimension of the embedding plot, either 2D or 3D, by setting ``_is_plot_3d``.

        If the embedding dimension is equal or higher to 3:

            * If ``idx_order`` is not provided, the plot will be 3D by default.
            * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if only 2 dimensions
                are provided, the plot will be 2D.

        If the embedding dimension is equal to 2:

            * If ``idx_order`` is not provided, the plot will be 2D by default.
            * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if 2 dimensions
                are provided, the plot will be 2D.

        This is supposing that the dimensions provided to ``idx_order`` are in the range of the number of
        dimensions of the embedding (i.e., between 0 and :py:attr:`cebra.CEBRA.output_dimension` -1).

        Args:
            embedding: A matrix containing the feature representation computed with CEBRA.
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).
        """
        if (idx_order is None and
                embedding.shape[1] == 2) or (idx_order is not None and
                                             len(idx_order) == 2):
            self._is_plot_3d = False
        elif (idx_order is None and
              embedding.shape[1] >= 3) or (idx_order is not None and
                                           len(idx_order) == 3):
            self._is_plot_3d = True
        else:
            raise ValueError(
                f"Invalid embedding dimension, expects 2D or more, got {self.embedding.shape[1]}"
            )

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]):
        """Define the ax on which to generate the plot.

        Args:
            axis: A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            if self._is_plot_3d:
                self.ax = self.fig.add_subplot(projection="3d")
            else:
                self.ax = self.fig.add_subplot()
        else:
            self.ax = axis

    def _define_idx_order(self, idx_order: Optional[Tuple[int]]) -> Tuple[int]:
        """Check that the index order has a valid number of dimensions compared to the number of
        dimensions of the embedding.

        Args:
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).

        Returns:
            The index order for the corresponding 2D or 3D plot.
        """

        if idx_order is None:
            if self._is_plot_3d:
                idx_order = (0, 1, 2)
            else:
                idx_order = (0, 1)
        else:
            # If the idx_order was provided by the user
            self._check_valid_dimensions(idx_order)
        return idx_order

    def _check_valid_dimensions(self, idx_order: Tuple[int]):
        """Check that provided dimensions are valid.

        The provided dimensions need to be 2 if the plot is set to a 2D plot and 3 if it is set to 3D.
        The dimensions values should be in the range of the embedding dimensionality.

        Args:
            idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
                embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
                dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
                (e.g., (2, 4, 5)).
        """
        # Check size validity
        if self._is_plot_3d and len(idx_order) != 3:
            raise ValueError(
                f"idx_order must contain 3 dimension values, got {len(idx_order)}."
            )
        elif not self._is_plot_3d and len(idx_order) != 2:
            raise ValueError(
                f"idx_order must contain 2 dimension values, got {len(idx_order)}."
            )

        # Check value validity
        for dim in idx_order:
            if dim < 0 or dim > self.embedding.shape[1] - 1:
                raise ValueError(
                    f"List of dimensions to plot is invalid, got {idx_order}, with {dim} invalid."
                    f"Values should be between 0 and {self.embedding.shape[1]}."
                )

    def _plot_3d(self,
                 grey_fig: bool = False,
                 **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding in 3d.

        Args:
            grey_fig: Set the title and edge color to grey, to be visible on both white and black
                backgrounds.


        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """

        idx1, idx2, idx3 = self.idx_order
        self.ax.scatter(
            xs=self.embedding[:, idx1],
            ys=self.embedding[:, idx2],
            zs=self.embedding[:, idx3],
            c=self.embedding_labels,
            cmap=self.cmap,
            alpha=self.alpha,
            s=self.markersize,
            **kwargs,
        )

        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("w")
        self.ax.yaxis.pane.set_edgecolor("w")
        self.ax.zaxis.pane.set_edgecolor("w")
        self.ax.set_title(self.title, y=1.0, pad=-10)

        if grey_fig:
            self.ax.xaxis.pane.set_edgecolor("grey")
            self.ax.yaxis.pane.set_edgecolor("grey")
            self.ax.zaxis.pane.set_edgecolor("grey")

        return self.ax

    def _plot_2d(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding in 2d.

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """

        idx1, idx2 = self.idx_order
        self.ax.scatter(
            x=self.embedding[:, idx1],
            y=self.embedding[:, idx2],
            c=self.embedding_labels,
            cmap=self.cmap,
            alpha=self.alpha,
            s=self.markersize,
            **kwargs,
        )

        return self.ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the embedding.

        Note:
            To set the entire figure to grey, you should add that snippet of code:

            >>> from matplotlib import rcParams
            >>> rcParams['xtick.color'] = 'grey'
            >>> rcParams['ytick.color'] = 'grey'
            >>> rcParams['axes.labelcolor'] = 'grey'
            >>> rcParams['axes.edgecolor'] = 'grey'
            >>> rcParams['axes.titlecolor'] = 'grey'

        Returns:
            The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.
        """
        if isinstance(self.embedding_labels, str):
            if self.embedding_labels == "time":
                self.embedding_labels = np.arange(self.embedding.shape[0])
            elif not matplotlib.colors.is_color_like(self.embedding_labels):
                raise ValueError(
                    f"Embedding labels invalid: provide a list of index or a valid str (time or valid colorname), got {self.embedding_labels}."
                )
            self.cmap = None
        elif isinstance(self.embedding_labels, Iterable):
            if len(self.embedding_labels) != self.embedding.shape[0]:
                raise ValueError(
                    f"Invalid embedding labels: the labels vector should have the same number of samples as the embedding, got {len(self.embedding_labels)}, expect {self.embedding.shape[0]}."
                )
            if self.embedding_labels.ndim > 1:
                raise NotImplementedError(
                    f"Invalid embedding labels: plotting does not support multiple sets of labels, got {self.embedding_labels.ndim}."
                )

        if self._is_plot_3d:
            self.ax = self._plot_3d(**kwargs)
        else:
            self.ax = self._plot_2d(**kwargs)
        if isinstance(self.ax, matplotlib.axes._axes.Axes):
            self.ax.set_title(self.title)

        return self.ax


def plot_embedding_cebra(
    embedding: Union[npt.NDArray, torch.Tensor],
    embedding_labels: Optional[Union[npt.NDArray, torch.Tensor, str]] = "grey",
    ax: Optional[matplotlib.axes.Axes] = None,
    idx_order: Optional[Tuple[int]] = None,
    markersize: float = 1,
    alpha: float = 0.4,
    cmap: str = "cool",
    title: str = "Embedding",
    figsize: Tuple[int] = (8, 8),
    dpi: float = 100,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot embedding in a 3D or 2D dimensional space.

    If the embedding dimension is equal or higher to 3:

        * If ``idx_order`` is not provided, the plot will be 3D by default.
        * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if only 2 dimensions are provided, the plot will be 2D.

    If the embedding dimension is equal to 2:

        * If ``idx_order`` is not provided, the plot will be 2D by default.
        * If ``idx_order`` is provided, if it has 3 dimensions, the plot will be 3D, if 2 dimensions are provided, the plot will be 2D.

    This is supposing that the dimensions provided to ``idx_order`` are in the range of the number of
    dimensions of the embedding (i.e., between 0 and :py:attr:`cebra.CEBRA.output_dimension` -1).

    The function makes use of :py:func:`matplotlib.pyplot.scatter` and parameters from that function can be provided
    as part of ``kwargs``.


    Args:
        embedding: A matrix containing the feature representation computed with CEBRA.
        embedding_labels: The labels used to map the data to color. It can be:

            * A vector that is the same sample size as the embedding, associating a value to each of the sample, either discrete or continuous.
            * A string, either `time`, then the labels while color the embedding based on temporality, or a string that can be interpreted as a RGB(A) color, then the embedding will be uniformly display with that unique color.
        ax: Optional axis to create the plot on.
        idx_order: A tuple (x, y, z) or (x, y) that maps a dimension in the data to a dimension in the 3D/2D
            embedding. The simplest form is (0, 1, 2) or (0, 1) but one might want to plot either those
            dimensions differently (e.g., (1, 0, 2)) or other dimensions from the feature representation
            (e.g., (2, 4, 5)).
        markersize: The marker size.
        alpha: The marker blending, between 0 (transparent) and 1 (opaque).
        cmap: The Colormap instance or registered colormap name used to map scalar data to colors. It will be ignored if `embedding_labels` is set to a valid RGB(A).
        title: The title on top of the embedding.
        figsize: Figure width and height in inches.
        dpi: Figure resolution.
        kwargs: Optional arguments to customize the plots. See :py:func:`matplotlib.pyplot.scatter` documentation for more
            details on which arguments to use.

    Returns:
        The axis :py:meth:`matplotlib.axes.Axes.axis` of the plot.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 5))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(X, y)
        CEBRA(max_iterations=10)
        >>> embedding = cebra_model.transform(X)
        >>> ax = cebra.plot_embedding(embedding, embedding_labels='time')

    """
    return _EmbeddingPlot(
        embedding=embedding,
        embedding_labels=embedding_labels,
        axis=ax,
        idx_order=idx_order,
        markersize=markersize,
        alpha=alpha,
        cmap=cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
    ).plot(**kwargs)


def plot_embedding(ds_name, model_name, experiment_name, transfer_outputs=True):

    df = pd.read_csv(f"{DESTINATION}/cebra_data/{ds_name}")
    behaviors = df.iloc[:, -1].values
    labels = behaviors

    ## create new directories to keep model outputs
    experiment_dir, experiment = experiment_name.split("/")
    experiment_path = f"{DESTINATION}/cebra_outputs/{experiment_dir}"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    model_output_path = f"{DESTINATION}/cebra_outputs/{experiment_name}"
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)

    if transfer_outputs:
        scp_commands = [
            f'scp alicia@flv-c2.mit.edu:{SOURCE}/{experiment_name}/{model_name}_embd.h5 .',
            f'scp alicia@flv-c2.mit.edu:{SOURCE}/{experiment_name}/{model_name}_loss.npz .',
            f'scp alicia@flv-c2.mit.edu:{SOURCE}/{experiment_name}/{model_name}_temp.npz .'
        ]
        os.chdir(f'{DESTINATION}/cebra_outputs/{experiment_name}')
        for command in scp_commands:
            subprocess.run(command, shell=True)

    file = h5py.File(f'{model_output_path}/{model_name}_embd.h5', 'r')
    embedding = file['embedding']
    print(embedding.shape)

    ### DIY plotting
    xs = embedding[:, 0]
    ys = embedding[:, 1]
    zs = embedding[:, 2]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(xs, ys, zs, c=labels, cmap='viridis')
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_zlabel('Latent 3')
    fig.colorbar(surf, ax=ax)
    plt.show()
    """
    ### cebra's default plotting
    embedding = embedding[:, :]
    print(embedding.shape)
    plot_embedding_cebra(embedding, embedding_labels=labels)
    plt.show()

    ### plot loss
    """
    losses = np.load(f'{model_output_path}/{model_name}_loss.npz')['numpy_array']
    plt.plot(losses)
    plt.title('training losses')
    plt.show()
    temperatures = np.load(f'{model_output_path}/{model_name}_temp.npz')['numpy_array']
    plt.plot(temperatures)
    plt.title('temperature')
    plt.show()
    """


def plot_trajectory(ds_name, model_name, experiment_name, reversal_events):

    model_output_path = f"{DESTINATION}/cebra_outputs/{experiment_name}"
    file = h5py.File(f'{model_output_path}/{model_name}_embd.h5', 'r')
    embedding = file['embedding']

    df = pd.read_csv(f"{DESTINATION}/cebra_data/{ds_name}")
    behaviors = df.iloc[:, -1].values

    trajectories = reversal_events['2022-07-15-12']
    print(trajectories)
    print(len(trajectories))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    

    for i in range(len(trajectories) - 1):

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # colormap = plt.cm.get_cmap('viridis', len(trajectories))

        trajectory = list(range(trajectories[i], trajectories[i+1]))
        print(f"trajectory: {trajectory}")

        x_coords = embedding[trajectory][:, 0]
        y_coords = embedding[trajectory][:, 1]
        z_coords = embedding[trajectory][:, 2]

        # color = colormap(i / len(trajectory))
        # line = plt.plot(x_coords, y_coords, color='r')
        ax = plot_embedding_cebra(embedding, embedding_labels=behaviors)
        line = Line3D(x_coords, y_coords, z_coords, color='r')
        ax.add_line(line)
        # ax.scatter(x_coords, y_coords, color='r')
        ax.scatter(x_coords, y_coords, z_coords, color='r')
        plt.show()

    """
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(x_coords, y_coords, z_coords, color='red')
    #line = Line3D(x_coords, y_coords, z_coords, color='blue')
    #ax.add_line(line)

    #colors = plt.cm.viridis(np.linspace(0, 1, len(x_coords)))
    #line.set_color(colors)

    #plt.show()
    """


if __name__ == "__main__":
    ds_name = "AVA-AVE-AVB/AVA-AVE-AVB_reversal-velocities_f10_10.csv"
    experiment_name = "AVA-AVE-AVB/AVA-AVE-AVB_reversal-velocities_f10_10"
    model_name = "learning_rate_0.0001_min_temperature_1_num_hidden_units_8_output_dimension_5"
    reversal_events = \
    {'2022-07-15-12': [0, 15, 20, 48, 53, 57, 80, 96, 98, 103, 126, 142, 155,
        170, 192, 204, 281, 292, 303, 320, 326, 333, 341, 348, 361, 371, 378,
        394, 409, 418, 441, 463, 470, 488, 508, 518, 537, 554, 576, 598],
        '2023-01-05-18': [601, 617, 627, 640, 682, 705, 737, 771, 810, 851,
            921, 925, 949, 1000, 1008, 1013, 1040, 1054, 1109, 1137, 1193,
            1242, 1251, 1286, 1331, 1365, 1415, 1450], '2023-01-09-08': [1491,
                1498, 1523, 1527, 1562, 1573, 1585, 1604, 1615, 1620, 1650,
                1653, 1656, 1666, 1678, 1697, 1700, 1729, 1739, 1776, 1781,
                1785, 1787, 1791, 1793, 1798, 1847, 1869, 1874, 1879, 1881,
                1911, 1917, 1933, 1936, 1939, 1945, 1957, 1982, 1986, 1990,
                1993, 1997, 2000, 2002, 2007, 2012, 2017, 2031, 2034, 2053,
                2055, 2078, 2080, 2082, 2085, 2093, 2096, 2104, 2110, 2129,
                2138, 2142, 2152, 2163, 2175, 2191], '2023-01-10-07': [2196,
                    2200, 2217, 2222, 2226, 2232, 2241, 2253, 2265, 2272, 2278,
                    2284, 2293, 2309, 2323, 2337, 2344, 2353, 2361, 2390, 2407,
                    2419, 2427, 2439, 2453, 2472, 2488, 2511, 2518, 2523, 2532,
                    2556, 2567, 2582, 2594, 2614, 2620, 2640, 2652, 2664, 2688,
                    2707, 2741, 2756, 2778, 2790, 2827], '2023-01-10-14':
                [2842, 2848, 2856, 2866, 2875, 2893, 2903, 2928, 2947, 2966,
                    2977, 2987, 3009, 3021, 3038, 3050, 3061, 3075, 3092, 3103,
                    3115, 3130, 3134, 3144, 3155, 3168, 3189, 3208, 3224, 3245,
                    3260, 3273, 3290, 3305, 3313, 3325, 3338, 3351, 3370, 3382,
                    3385], '2023-01-16-08': [3404, 3409, 3427, 3436, 3448,
                        3459, 3477, 3525, 3528, 3581, 3592, 3601, 3605, 3623,
                        3639, 3653, 3656, 3666, 3681, 3686, 3696, 3702, 3710,
                        3725, 3730, 3734, 3743, 3755, 3807, 3818, 3822, 3869,
                        3911, 3947, 3966, 4010, 4054, 4065, 4078, 4086, 4100,
                        4110, 4125, 4146, 4167, 4180, 4186, 4192, 4202],
                    '2023-01-19-01': [4207, 4212, 4214, 4222, 4234, 4238, 4244,
                        4246, 4261, 4266, 4272, 4276, 4289, 4298, 4310, 4313,
                        4317, 4329, 4335, 4342, 4344, 4363, 4367, 4384, 4392,
                        4409, 4419, 4421, 4424, 4434, 4436, 4441, 4450, 4476,
                        4494, 4512, 4534, 4556, 4575, 4579, 4588, 4606, 4625,
                        4633, 4640, 4652, 4663, 4670, 4682, 4698, 4705, 4735,
                        4741, 4763, 4766, 4780, 4815], '2023-01-19-08': [4830,
                            4833, 4839, 4844, 4846, 4851, 4858, 4862, 4868,
                            4872, 4877, 4881, 4886, 4893, 4895, 4899, 4908,
                            4916, 4920, 4928, 4940, 4943, 4946, 4948, 4965,
                            4967, 4971, 4981, 4989, 4997, 5003, 5005, 5021,
                            5032, 5039, 5043, 5046, 5052, 5057, 5060, 5072,
                            5078, 5082, 5100, 5102, 5108, 5117, 5142, 5146,
                            5164, 5166, 5169, 5173, 5188, 5190, 5206, 5218,
                            5221, 5223, 5226, 5229, 5231, 5233, 5237, 5240,
                            5246, 5252, 5269, 5289, 5292, 5296, 5316, 5322,
                            5334, 5339, 5349, 5351, 5362, 5374, 5378],
                        '2023-01-19-15': [5385, 5391, 5396, 5409, 5419, 5424,
                            5438, 5452, 5463, 5469, 5471, 5480, 5487, 5499,
                            5510, 5522, 5542, 5553, 5562, 5569, 5572, 5586,
                            5603, 5617, 5625, 5630, 5640, 5650, 5658, 5666,
                            5676, 5689, 5694, 5698, 5708, 5731, 5736, 5746,
                            5751, 5769, 5781, 5793, 5800, 5809, 5811, 5815,
                            5822, 5828, 5843, 5852, 5863, 5884, 5886, 5891,
                            5901, 5906, 5915, 5925, 5930, 5947, 5952, 5956,
                            5963, 5965, 5977, 5979], '2023-01-23-15': [5999,
                                    6008, 6015, 6032, 6046, 6054, 6069, 6074,
                                    6103, 6128, 6157, 6186, 6188, 6193, 6198,
                                    6211, 6214, 6216, 6225, 6240, 6249, 6282,
                                    6290, 6296, 6303, 6321, 6332, 6349, 6364,
                                    6386, 6399, 6415, 6447, 6456, 6469, 6488,
                                    6500, 6518, 6539, 6550, 6569]}

    # plot_embedding(ds_name, model_name, experiment_name, False)
    plot_trajectory(ds_name, model_name, experiment_name, reversal_events)
