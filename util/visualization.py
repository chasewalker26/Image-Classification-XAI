import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def _normalize_scale(attr, scale_factor):
    attr_norm = attr / scale_factor

    return np.clip(attr_norm, -1, 1)

def _cumulative_sum_threshold(values, percentile):
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_attr(attr, norm, outlier_perc = 2, reduction_axis = 2):
    attr_combined = np.sum(attr, axis = reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if norm == "absolute":
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif norm == "positive":
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif norm == "negative":
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif norm == "all":
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)

    return _normalize_scale(attr_combined, threshold)

def attr_to_subplot(input, title, axs, norm = "absolute", cmap = None, original_image = False, blended_image = None, alpha = 0.5):
    axs.set_title(title)
    if original_image == False:
        if norm == "absolute":
            norm_cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, (1, 1, 1, 1)), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)   
            # norm_cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, (1, 1, 1, 1)), (0.25, (0, 0, 1, 0.25)), (0.5, (0, 0, 1, 0.5)), (0.75, (0, 0, 1, 0.75)), (1, (0, 0, 1, 1))], N = 256)   
            # # norm_cmap = LinearSegmentedColormap.from_list('custom blue',  
            #                                               [(0.0, (1, 1, 1, 0)), 
            #                                                (0.01, (0, 0, 1, 0.1)), 
            #                                                (0.11, (0, 0, 1, 0.2)), 
            #                                                (0.2, (0, 0, 1, 0.3)), 
            #                                                (0.3, (0, 0, 1, 0.4)), 
            #                                                (0.4, (0, 0, 1, 0.5)), 
            #                                                (0.5, (0, 0, 1, 0.6)), 
            #                                                (0.6, (0, 0, 1, 0.7)), 
            #                                                (0.7, (0, 0, 1, 0.8)), 
            #                                                (0.8, (0, 0, 1, 0.9)), 
            #                                                (0.9, (0, 0, 1, 0.95)), 
            #                                                (1.0, (0, 0, 1, 1))
            #                                                ], N = 256)   
            vmin, vmax = 0, 1
        elif norm == "positive":
            norm_cmap = "Greens"
            vmin, vmax = 0, 1
        elif norm == "negative":
            norm_cmap = "Reds"
            vmin, vmax = 0, 1
        elif norm == "all":
            norm_cmap = LinearSegmentedColormap.from_list("RdGn", ["red", "white", "blue"])
            vmin, vmax = -1, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")

        if cmap is None:
            cmap = norm_cmap

        if blended_image is not None:
            axs.imshow(np.transpose(blended_image.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            axs.imshow(_normalize_attr(input, norm), cmap = cmap, vmin = vmin, vmax = vmax, alpha = alpha)
        else:
            axs.imshow(_normalize_attr(input, norm), cmap = cmap, vmin = vmin, vmax = vmax)

    else:
        # Only show input image
        axs.imshow(np.transpose(input.squeeze().detach().cpu().numpy(), (1, 2, 0)))

    axs.set_xticks([])
    axs.set_yticks([])

    return