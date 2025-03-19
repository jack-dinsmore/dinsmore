import numpy as np

def step(ax, x_edges, y, **kwargs):
    if len(x_edges) != len(y) + 1:
        raise Exception(f"Length of x ({len(x_edges)}) must be one greater than length of y ({len(y)})")
    new_y = np.concatenate([[y[0]], y])
    ax.step(x_edges, new_y, **kwargs)



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
    handles = np.array(handles)
    labels = np.array(labels)

    # These two lines set unique_labels equal to the output of np.unique, but rearranged so that
    # they are in the same order as the original list
    _, idx = np.unique(labels, return_index=True)
    unique_labels = labels[np.sort(idx)]

    out_handles = []
    for label in unique_labels:
        out_handles.append(tuple(handles[labels==label]))
        
    return out_handles, list(unique_labels)