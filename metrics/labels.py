import numpy as np
import itertools
from skimage.segmentation import find_boundaries
from skimage.metrics import hausdorff_distance
from sklearn.metrics import pairwise_distances_argmin_min
from .utils import check_broadcastable


__all__ = ['hausdorff', 'hausdorff_boundary', 'min_distance', 'min_distance_boundary']


def min_distance(pred, ref, labels=None, reduction='mean',
                 label_reduction='mean', inf=0):
    """Compute the minimum distance between each boundary point in ref
    and any point in pred.

    When ``reduction=max``, this is equivalent to ``hausdorff``.

    Args:
        pred (ndarray): Predicted segmentation
        ref (ndarray): Reference segmentation
        labels (list[int]): Labels to include.
        reduction ({'mean', 'sum', 'max', 'min'}): Reduction across ref points.
        label_reduction ({'mean', 'weighted', 'sum', 'max', None}): Reduction across labels
        inf (float): Value to use for infinite distances

    Returns:
        dist (float or dict{int: float}):
            Minimum distance (per label if `label_reduction=None`)

    """
    check_broadcastable(pred, ref)
    def default_labels():
        l1 = np.unique(ref).tolist()
        l2 = np.unique(pred).tolist()
        l = set(l1).union(set(l2))
        if 0 in l:
            l.remove(0)
        return l
    labels = labels or default_labels()
    coord = np.stack(np.meshgrid(*[np.arange(s) for s in pred.shape],
                                 indexing='ij'), axis=-1)
    dist = {}
    for l in labels:
        x = find_boundaries(pred == l, mode='inner')
        y = find_boundaries(ref == l, mode='inner')
        ycount = y.sum()
        xcount = x.sum()
        if ycount == 0:
            dist[l] = 0
            continue
        elif xcount == 0:
            dist[l] = inf
            continue
        x = coord[x, :]
        y = coord[y, :]
        _, d = pairwise_distances_argmin_min(y, x)
        if reduction == 'sum':
            d = d.sum()
        elif reduction == 'mean':
            d = d.sum() / d.size
        elif reduction == 'min':
            d = d.min()
        elif reduction == 'max':
            d = d.max()
        elif reduction == 'median':
            d = np.median(d)
        elif callable(reduction):
            d = reduction(d)
        else:
            raise ValueError(f'Don\'t know what to do with reduction '
                             f'{reduction}')
        if d == float('inf'):
            d = inf
        dist[l] = d
    if label_reduction == 'sum':
        dist = sum(dist.values())
    elif label_reduction == 'mean':
        dist = sum(dist.values()) / len(dist)
    elif label_reduction == 'max':
        dist = max(dist.values())
    return dist


def hausdorff(pred, ref, labels=None, reduction='mean', symmetric=True, inf=0):
    """Compute the Hausdorff distance between two segmentations.

    Assuming x and y are two binary volumes, the Hausdorff distance is:
        H(x, y) = max_i min_j ||x_j - y_i||^2
    The symmetric version is:
        H_sym(x, y) = max(H(x, y), H(y, x))
    In the multilabel case, the distance is computed per label, then reduced:
        H_multi = reduce_l H(x == l, y == l)
    The label 0 is not included by default. If a weighted reduction is used,
    the weights are the label volume in the reference image. In the
    symmetric case, it is the average label volume across both images.


    Args:
        pred (ndarray): Predicted segmentation
        ref (ndarray): Reference segmentation
        labels (list[int]): Labels to include.
        reduction ({'mean', 'weighted', 'sum', 'max', None}): Multilabel reduction
        symmetric (bool): Use symmetric distance
        inf (float): Value to use for infinite distances

    Returns:
        dist (float or dict{int: float}):
            Distance (per label if `reduction=None`)

    """
    check_broadcastable(pred, ref)
    def default_labels():
        l1 = np.unique(ref).tolist()
        l2 = np.unique(pred).tolist()
        l = set(l1).union(set(l2))
        if 0 in l:
            l.remove(0)
        return l
    labels = labels or default_labels()
    dist = {}
    weights = {}
    for l in labels:
        x = pred == l
        y = ref == l
        h = hausdorff_distance(y, x)
        if symmetric:
            h = max(h, hausdorff_distance(x, y))
        if h == float('inf'):
            h = inf
        dist[l] = h
        if reduction == 'weighted':
            w = y.sum()
            if symmetric:
                w = (w + x.sum()) / 2
            weights[l] = w
    if reduction == 'sum':
        dist = sum(dist.values())
    elif reduction == 'mean':
        dist = sum(dist.values()) / len(dist)
    elif reduction == 'max':
        dist = max(dist.values())
    elif reduction == 'weighted':
        dist = sum([h*w for h, w in zip(dist.values(), weights.values())]) / sum(weights.values())
    return dist


def hausdorff_boundary(pred, ref, labels=None, reduction='mean',
                       symmetric=True, inf=0, thick=False):
    """Hausdorff distance between pairwise boundaries.

    This function computes the Hausdorff distance between label pairs.
    While ``hausdorff`` computes the (interior) boundary of each label,
    ``hausdorff_boundary`` extracts the boundary between specific pairs
    of labels.

    Note that the weighted version therefore weights by boundary length,
    instead of label volume.

    Args:
        pred (ndarray): Predicted segmentation
        ref (ndarray): Reference segmentation
        labels (list[int]): Labels to include.
        reduction ({'mean', 'weighted', 'sum', 'max', None}): Multilabel reduction
        symmetric (bool): Use symmetric distance
        inf (float): Value to use for infinite distances
        thick (bool): Use a thick boundary, instead of a symmetric inner one.

    Returns:
        dist (float or dict{frozenset[int]: float}):
            Distance (per label pair if `reduction=None`)
    """
    check_broadcastable(pred, ref)

    def default_labels():
        l1 = np.unique(ref).tolist()
        l2 = np.unique(pred).tolist()
        l = set(l1).union(set(l2))
        return l

    def boundaries(l1, l2):
        if thick:
            bound = [np.bitwise_and(find_boundaries(l1, mode='thick'),
                                    find_boundaries(l2, mode='thick'))]
        else:
            bound = [np.bitwise_and(find_boundaries(l1, mode='inner'),
                                    find_boundaries(l2, mode='outer')),
                     np.bitwise_and(find_boundaries(l2, mode='inner'),
                                    find_boundaries(l1, mode='outer'))]
        return bound

    labels = labels or default_labels()

    dist = {}
    weights = {}
    for l1, l2 in itertools.combinations(labels, 2):
        x = boundaries(pred == l1, pred == l2)
        y = boundaries(ref == l1, ref == l2)
        h = max(hausdorff(xx, yy, symmetric=symmetric, inf=inf)
                for xx, yy in zip(x, y))
        dist[frozenset([l1, l2])] = h
        if reduction == 'weighted':
            w = sum(yy.sum() for yy in y) / len(y)
            if symmetric:
                w = (w + sum(xx.sum() for xx in x) / len(x)) / 2
            weights[frozenset([l1, l2])] = w
    if reduction == 'sum':
        dist = sum(dist.values())
    elif reduction == 'mean':
        dist = sum(dist.values()) / len(dist)
    elif reduction == 'max':
        dist = max(dist.values())
    elif reduction == 'weighted':
        dist = sum([h*w for h, w in zip(dist.values(), weights.values())]) / sum(weights.values())
    return dist


def min_distance_boundary(pred, ref, labels=None, reduction='mean',
                          label_reduction='mean', inf=0, thick=False):
    """Hausdorff distance between pairwise boundaries.

    This function computes the Hausdorff distance between label pairs.
    While ``hausdorff`` computes the (interior) boundary of each label,
    ``hausdorff_boundary`` extracts the boundary between specific pairs
    of labels.

    Note that the weighted version therefore weights by boundary length,
    instead of label volume.

    Args:
        pred (ndarray): Predicted segmentation
        ref (ndarray): Reference segmentation
        labels (list[int]): Labels to include.
        reduction ({'mean', 'sum', 'min', 'max', None}): Reduction across points
        label_reduction ({'mean', 'sum', 'max', None}): Reduction across labels
        inf (float): Value to use for infinite distances
        thick (bool): Use a thick boundary, instead of a symmetric inner one.

    Returns:
        dist (float or dict{frozenset[int]: float}):
            Distance (per label pair if `reduction=None`)
    """
    check_broadcastable(pred, ref)

    def default_labels():
        l1 = np.unique(ref).tolist()
        l2 = np.unique(pred).tolist()
        l = set(l1).union(set(l2))
        return l

    def boundaries(l1, l2):
        if thick:
            bound = [np.bitwise_and(find_boundaries(l1, mode='thick'),
                                    find_boundaries(l2, mode='thick'))]
        else:
            bound = [np.bitwise_and(find_boundaries(l1, mode='inner'),
                                    find_boundaries(l2, mode='outer')),
                     np.bitwise_and(find_boundaries(l2, mode='inner'),
                                    find_boundaries(l1, mode='outer'))]
        return bound

    labels = labels or default_labels()

    dist = {}
    weights = {}
    for l1, l2 in itertools.combinations(labels, 2):
        x = boundaries(pred == l1, pred == l2)
        y = boundaries(ref == l1, ref == l2)
        h = max(min_distance(xx, yy, inf=inf, reduction=reduction)
                for xx, yy in zip(x, y))
        dist[frozenset([l1, l2])] = h
    if label_reduction == 'sum':
        dist = sum(dist.values())
    elif label_reduction == 'mean':
        dist = sum(dist.values()) / len(dist)
    elif label_reduction == 'max':
        dist = max(dist.values())
    elif label_reduction == 'weighted':
        dist = sum([h*w for h, w in zip(dist.values(), weights.values())]) / sum(weights.values())
    return dist

