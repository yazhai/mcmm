import typing
import numpy.typing as npt

import numpy as np


def partition_space(
    lb: typing.List[float],
    ub: typing.List[float],
    xs: npt.ArrayLike,
    dim: int,
    ignore_dist: float = 0,
) -> typing.List[typing.Tuple]:
    assert xs.shape[1] == len(lb) == len(ub), "Dimensionality mismatch"
    assert 0 <= dim < xs.shape[1], "Invalid dimension"

    # sort xs based on the specified dimension
    idx_unsort_to_sorted = xs[:, dim].argsort()
    xs_sorted = xs[idx_unsort_to_sorted]

    # calculate the midpoints between adjacent points in the specified dimension
    midpoints = np.mean(np.vstack((xs_sorted[:-1, dim], xs_sorted[1:, dim])), axis=0)

    # Form boundaries of the partitioned dimension
    boundaries = [lb[dim]] + midpoints.tolist() + [ub[dim]]

    # Populate boxed by modifying the dimension of interest
    partitioned_boxes_sorted = []
    for idx in range(1, len(boundaries)):
        box_lb = lb.copy()
        box_ub = ub.copy()
        box_lb[dim] = boundaries[idx - 1]
        box_ub[dim] = boundaries[idx]
        partitioned_boxes_sorted.append((box_lb, box_ub))

    # Reorder the boxes to match the original order of xs
    idx_sorted_to_unsort = np.argsort(idx_unsort_to_sorted)
    partitioned_boxes = []
    for idx in idx_sorted_to_unsort:
        partitioned_boxes.append(partitioned_boxes_sorted[idx])

    return partitioned_boxes


if __name__ == "__main__":
    lb = [0, 0, 0]
    ub = [1, 1, 1]
    xs = np.random.rand(10, 3)
    dim = 1
    print("Before partitioning: ")
    print("lb: (" + str(lb[0]) + ", " + str(lb[1]) + ", " + str(lb[2]) + ")")
    print("ub: (" + str(ub[0]) + ", " + str(ub[1]) + ", " + str(ub[2]) + ")")
    print("xs: ")
    print(np.around(xs, decimals=2))
    print()

    print("After partitioning on dim {}: ".format(dim))
    partitioned_boxes = partition_space(lb, ub, xs, dim)
    for idx, box in enumerate(partitioned_boxes):
        print("Box {}: ".format(idx))
        # print lb and ub of the box, rounded to 2 decimal place
        print(
            "lb: ("
            + str(np.around(box[0][0], decimals=2))
            + ", "
            + str(np.around(box[0][1], decimals=2))
            + ", "
            + str(np.around(box[0][2], decimals=2))
            + ")"
        )
        print(
            "ub: ("
            + str(np.around(box[1][0], decimals=2))
            + ", "
            + str(np.around(box[1][1], decimals=2))
            + ", "
            + str(np.around(box[1][2], decimals=2))
            + ")"
        )
