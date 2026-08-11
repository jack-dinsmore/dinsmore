import numpy as np
from palettable.cubehelix import Cubehelix
SEPIA = "#ba9988"
TURQUOISE = "#6baba5"
ORANGE = "#f28159"
SALMON = "#d98b88"
# https://rufflewind.com/_urandom/colorpicker/#ba9988

def cubehelix(start, rotations=0, hue_start=2, hue_end=2, gamma=1, lightness_start=0, lightness_end=1, reverse=False):
    # https://davidjohnstone.net/cubehelix-gradient-picker

    if np.abs(lightness_end - lightness_start) < 0.01:
        lightness_end = lightness_start + 0.01
    rotations /= (lightness_end - lightness_start)

    return Cubehelix.make(
        start=start/120+1, rotation=rotations, gamma=gamma,
        min_sat=hue_start, max_sat=hue_end, min_light=lightness_start, max_light=lightness_end,
        reverse=reverse, n=256
    ).get_mpl_colormap()

def step(ax, x_edges, y, **kwargs):
    if len(x_edges) != len(y) + 1:
        raise Exception(f"Length of x ({len(x_edges)}) must be one greater than length of y ({len(y)})")
    new_y = np.concatenate([[y[0]], y])
    ax.step(x_edges, new_y, **kwargs)

def step_between(ax, x_edges, y_top, y_bottom, **kwargs):
    xs = []
    y_tops = []
    y_bottoms = []
    for i in range(len(x_edges)-1):
        xs.append(x_edges[i])
        xs.append(x_edges[i+1])
        y_tops.append(y_top[i])
        y_tops.append(y_top[i])
        y_bottoms.append(y_bottom[i])
        y_bottoms.append(y_bottom[i])
    ax.fill_between(xs, y_tops, y_bottoms, **kwargs)

def vstep(ax, y_edges, x, **kwargs):
    """Make a vertical histogram"""
    if len(y_edges) != len(x) + 1:
        raise Exception(f"Length of y ({len(y_edges)}) must be one greater than length of x ({len(x)})")
    y_points = np.repeat(y_edges, 2)[1:-1]
    x_points = np.repeat(x, 2)
    ax.plot(x_points, y_points, **kwargs)

def diagram_arrow(ax, start, end, tilt=0.4, aspect=0.3, head_scale=0.05, color='k', lw=1, line_kwargs={}, arrow_kwargs={}):
    if tilt < 0 or tilt > 1:
        raise Exception("Tilt should be between 0 and 1")
    a_cubic = 16*tilt
    b_cubic = 1 - 4 * tilt
    local_xs = np.linspace(0, 1, 100)
    local_ys = a_cubic * (local_xs - 0.5)**3 + b_cubic * (local_xs - 0.5) + 0.5
    local_xs *= aspect

    # Come up with an orthogonal matrix that takes (0, 0) to start and (aspect, 1) to end
    delta = [end[0] - start[0], end[1] - start[1]]
    a = -((-aspect * delta[0] + delta[1])/(1 + aspect**2))
    b = -((-delta[0] - aspect * delta[1])/(1 + aspect**2))
    c = -((aspect * delta[0] - delta[1])/(1 + aspect**2))
    mat = np.array([
        [a, b],
        [b, c],
    ])
    hypotenuse = np.sqrt(delta[0]**2 + delta[1]**2)

    global_xs, global_ys = np.transpose(np.transpose(np.einsum("ij,jk->ik", mat, [local_xs, local_ys])) + start)

    ax.plot(global_xs, global_ys, color=color, lw=lw, **line_kwargs)
    ax.arrow(global_xs[-1], global_ys[-1], global_xs[-1] - global_xs[-2], global_ys[-1] - global_ys[-2], color=color, width=0, length_includes_head=True, head_width=hypotenuse*head_scale, overhang=0.3, **arrow_kwargs)

def crunch_legend(ax):
    """
    Takes the handles and labels of the current axis, combines all handles with the same label into
    one line, and then returns the resulting handles and labels. Order is preserved. 

    You can create the legend with e.g. fig.legend(handles, labels, ncol=...)
    """
    handles, labels = ax.get_legend_handles_labels()
    handles = np.array(handles, dtype="object")
    labels = np.array(labels)

    # These two lines set unique_labels equal to the output of np.unique, but rearranged so that
    # they are in the same order as the original list
    _, idx = np.unique(labels, return_index=True)
    unique_labels = labels[np.sort(idx)]

    out_handles = []
    for label in unique_labels:
        out_handles.append(tuple(handles[labels==label]))
        
    return out_handles, list(unique_labels)


    
def fix_float(f, thresh=1e-10):
    if np.abs(f - np.round(f)) < thresh:
        return np.round(f)
    else:
        return f

def get_natural_ticks_sexa(lim, sepn=None, target_len=4):
    """Returns some ticks which fall within lim. If not provided, the separation between ticks is chosen to be a round number and be close to `target_len` in length"""
    if sepn is None:
        rng = np.abs(lim[1] - lim[0])
        try_power = int(np.log(rng)/np.log(60) - 1)

        # Find the power with the closest to four ticks
        best_len = np.inf
        best_ticks = None
        for power in range(try_power, try_power+3):
            for sepn in [1, 2, 5, 10, 15, 30]:
                ticks = get_natural_ticks_sexa(lim, 60**power * sepn)
                if np.abs(best_len - target_len) > np.abs(len(ticks) - target_len):
                    best_len = len(ticks)
                    best_ticks = ticks
        return best_ticks
    
    else:
        # Generate ticks with this sepn
        scale_lim = (lim[0] / sepn, lim[1] / sepn)
        # List all integers between low and high
        ticks = np.arange(max(*scale_lim) - min(*scale_lim)).astype(float)
        ticks += np.ceil(min(*scale_lim))
        ticks *= sepn
        ticks = ticks[(ticks >= min(*lim)) & (ticks <= max(*lim))]
        return ticks

def replace_labels_sexa(ax, center_pos):
    """Assumes the current labels are deviations in arcsec and sets the new labels to be in
    sexagesimal format, given center_pos as the ra, dec of (0, 0) in degrees.
    """
    stretch = np.cos(center_pos[1] * np.pi / 180)

    ticks = get_natural_ticks_sexa((np.array(ax.get_xlim())/stretch + center_pos[0]*3600)/15)
    ticks = np.flip(ticks)
    ticklabels = []
    old_sh = None
    old_sm = None
    old_ss = None
    for tick in ticks:
        h = fix_float(np.abs(tick / 3600))
        m = fix_float((h - int(h)) * 60)
        s = fix_float((m - int(m)) * 60)

        sh = f"{int(h):02d}$^\\mathrm{{h}}$"
        if sh == old_sh: sh = ""
        else: old_sh = sh
        sm = f"{int(m):02d}$^\\mathrm{{m}}$"
        if sm == old_sm: sm = ""
        else: old_sm = sm
        ss = f"{int(s):02d}$^\\mathrm{{s}}$"
        if ss == old_ss or (sm != "" and s == 0.): ss = ""
        else: old_ss = ss

        ticklabels.append(f"{sh}{sm}{ss}")
    ax.set_xticks((ticks*15 - center_pos[0]*3600) * stretch)
    ax.set_xticklabels(ticklabels)

    ticks = get_natural_ticks_sexa(np.array(ax.get_ylim()) + center_pos[1]*3600)
    ticks = np.flip(ticks)
    ticklabels = []
    old_sd = None
    old_sm = None
    old_ss = None
    for tick in ticks:
        if tick < 0:
            sign = "-"
        else:
            sign = ""
        d = fix_float(np.abs(tick / 3600))
        m = fix_float((d - int(d)) * 60)
        s = fix_float((m - int(m)) * 60)

        sd = f"{sign}{int(d):02d}$^\\circ$"
        if sd == old_sd: sd = ""
        else:
            old_sd = sd
            old_sm = None
            old_ss = None
        sm = f"{int(m):02d}$'$"
        if sm == old_sm:
            sm = ""
        else:
            old_sm = sm
            old_ss = None
        ss = f"{int(s):02d}$''$"
        if ss == old_ss:
            ss = ""
        else:
            old_ss = ss

        ticklabels.append(f"{sd}{sm}{ss}")

    ax.set_yticks(ticks - center_pos[1]*3600)
    ax.set_yticklabels(ticklabels)

def explicit_clip(fig):
    # Attempt to fix the plotted stuff escaping the plot, which Stefano was seeing in his printed versions
    for ax in fig.axes:
        for artist in ax.get_children():
            artist.set_clip_path(ax.patch)

def broken_plot(ax, x, y, break_distance=90, **kwargs):
    deltas = np.abs(y[1:] - y[:-1])
    step = 2*break_distance
    break_indices = np.where(deltas > break_distance)[0]
    break_indices = np.concatenate([[0], break_indices, [len(deltas)-1]])
    for i in range(len(break_indices)-1):
        start = break_indices[i]
        stop = break_indices[i+1]+2
        new_start = y[start] + step if y[start] < y[start+1] else y[start] - step
        if stop == len(y):
            stop -= 1
            newy = np.concatenate([[new_start], y[start+1:stop]])
        else:
            new_stop = y[stop] + step if y[stop] < y[stop+1] else y[stop] - step
            newy = np.concatenate([[new_start], y[start+1:stop-1], [new_stop]])

        ax.plot(x[start:stop], newy, **kwargs)
