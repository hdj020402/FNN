"""Visualization utilities — scatter plots, loss curves, model evaluation plots."""
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Literal
from numpy.typing import ArrayLike
from functools import partial

from utils.gen_model import gen_model
from utils.evaluation import Evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def scatter(
    *args: list[ArrayLike],
    scatter_label: list[str],
    output_path: str,
    s: ArrayLike | float = 10.0,
    marker: list[Literal['o', 'v', '^', 's', 'x', 'D']] = None,
    text: str = None,
    dot_color_list: list[str] = None,
    line: bool = True,
    line_color: str = 'r',
    line_width: float = 0.5,
    xlabel: str = 'Target',
    ylabel: str = 'Predict',
    figsize: tuple = (10, 10),
    axis_fontsize: float = 14,
    ylabel_fontsize: float = 16,
    xlabel_fontsize: float = 16,
    legend_fontsize: float = 10,
    bbox_to_anchor: tuple[float, float] = (1.0, 1.0),
    label_fontweight: Literal['normal', 'bold'] = 'normal',
):
    """Scatter plot of predictions vs targets."""
    if marker is None:
        marker = ['o', 'o', 'o', 'o']
    if dot_color_list is None:
        dot_color_list = ['#03788C', '#F27457', '#03488C', 'indianred', 'steelblue']

    plt.rcParams['font.size'] = axis_fontsize
    plt.rcParams['mathtext.fontset'] = 'custom'

    fig = plt.figure(figsize=figsize, dpi=300)
    ax_1 = fig.add_subplot(111)
    ax_1.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight=label_fontweight)
    ax_1.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight=label_fontweight)

    lower_limit = np.inf
    upper_limit = -np.inf
    for i, data in enumerate(args):
        x = data[0].flatten().cpu().detach().numpy() if isinstance(data[0], torch.Tensor) else data[0]
        y = data[1].flatten().cpu().detach().numpy() if isinstance(data[1], torch.Tensor) else data[1]
        lower_limit = min([min(x), min(y)]) if min([min(x), min(y)]) < lower_limit else lower_limit
        upper_limit = max([max(x), max(y)]) if max([max(x), max(y)]) > upper_limit else upper_limit
        ax_1.scatter(x, y, s, dot_color_list[i], label=scatter_label[i], marker=marker[i])

    if line:
        displacement = (upper_limit - lower_limit) * 0.1
        ax_1.plot(
            [lower_limit - displacement, upper_limit + displacement],
            [lower_limit - displacement, upper_limit + displacement],
            color=line_color,
            linewidth=line_width,
            label='y = x',
        )
    fig.legend(bbox_to_anchor=bbox_to_anchor, bbox_transform=ax_1.transAxes, fontsize=legend_fontsize)

    if text:
        plt.text(lower_limit, upper_limit, f'{text}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Loss-epoch curve
# ═══════════════════════════════════════════════════════════════════════════════

def loss_epoch(
    data_list: list[list[list[float]]],
    label_list: list[str],
    color_list: list[str],
    xlabel: str,
    ylabel: str,
    output_path: str,
    fontsize: float = 20,
):
    """Line graph for loss-vs-epoch curves."""
    assert len(data_list) == len(label_list), 'Data_list and Label_list are not of the same length.'
    assert len(data_list) <= len(color_list), 'There are not enough colors.'
    plt.figure(figsize=(10, 10), dpi=300)
    for i, data in enumerate(data_list):
        x = data[0]
        y = data[1]
        plt.plot(x, y, color=color_list[i], label=label_list[i])
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter from saved model
# ═══════════════════════════════════════════════════════════════════════════════

def scatterFromModel(model_path: str, param, DATA, output_dir: str):
    """Load a saved model and generate scatter plots on train/val/test sets."""
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, DATA.dataset)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        return
    model.load_state_dict(checkpoint['model'])
    model.eval()

    eval_class = partial(
        Evaluation,
        model=model,
        device=device,
        norm_dict=DATA.norm_dict,
        transform=param.target_transform,
    )
    train_eval = eval_class(train_loader)
    val_eval = eval_class(val_loader)
    test_eval = eval_class(test_loader)
    eval_dict = {'train': train_eval, 'val': val_eval, 'test': test_eval}

    file_name = os.path.splitext(os.path.basename(model_path))[0]
    for key, value in eval_dict.items():
        for target, pred, task in zip(
            torch.split(value.target, 1, dim=-1),
            torch.split(value.pred, 1, dim=-1),
            param.target_list,
        ):
            scatter(
                [target, pred],
                scatter_label=[key],
                output_path=os.path.join(output_dir, f'{task}/{file_name}_{task}_{key}.png'),
            )
