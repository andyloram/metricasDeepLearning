import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def resize_single_image(image, target_size):
    import numpy as np
    from skimage import transform
    new_rows, new_cols = target_size[0:2]
    rows, columns = image.shape[0:2]
    orig_shape = [rows, columns]
    aux = transform.resize(np.array(image), (new_rows, new_cols))
    return aux, orig_shape


def stratified_cv_split_ttv_age_and_sex(age: np.array, sex: np.array, k: int, original_idx: np.array = None,
                                        age_bin_width: int = 2) -> dict:
    """
        Generate indices for k-fold Cross Validation by keeping the distribution of the vector age and sex in train,
        validation and test sets
        :param age: The vector of the first magnitude
        :param sex: The vector of the second magnitude
        :param k: The number of Cross Validation Folds/iterations
        :param original_idx: If each sample has an identifier, return it instead of the position of the sample
        :param age_bin_width: Width of the bin used to compile the age histogram
        :return: The train/validation/test indices for each Cross Validation iteration
        """
    if len(age) != len(sex):
        raise Exception("age and sex vectors must have the same length")
    if original_idx is not None and len(original_idx) != len(age):
        raise Exception("If set, original_idx must be equal length as age and sex vectors")

    import itertools

    if sex.dtype != np.float:
        sex = np.array([1.0 if s == "V" else 0.0 for s in sex])

    N = len(age)
    max_age = np.max(age) + 2
    age_bins = np.array([age_bin_width * i for i in range(int(max_age // age_bin_width) + 1)])
    histogram = np.histogram2d(age, sex, bins=[age_bins, 2])

    fold_idx = list(range(k))
    folds = {k: [] for k in fold_idx}
    fold_order = np.concatenate([np.random.permutation(fold_idx) for _ in range(int(np.ceil(N / k)))])
    positions_to_remove = len(fold_order) - N
    if positions_to_remove > 0:
        fold_order = fold_order[:-positions_to_remove]
    histogram_coordinates = list(itertools.product(np.arange(histogram[0].shape[0]), np.arange(histogram[0].shape[1])))
    next_histogram_coordinate = 0
    added = 0  # Numero de imaxes engadidas a un fold. Serve tamen para saber o indice da proxima mostra a engadir
    while added < N:
        ab, sb = histogram_coordinates[next_histogram_coordinate]
        idx_inside_bins = np.array(
            [i for i in range(N) if age_bins[ab] <= age[i] < age_bins[ab + 1] and sex[i] == sb])

        for i in idx_inside_bins:
            folds[fold_order[added]].append(i)
            added += 1
        next_histogram_coordinate += 1

    return folds
