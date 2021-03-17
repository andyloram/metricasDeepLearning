import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

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

def print_metrics(age, age_pred, sex, sex_pred, print_file):
    n_age = age.cpu().numpy()
    n_age_pred = age_pred.cpu().numpy()
    n_sex = sex.cpu().numpy()
    n_sex_pred = sex_pred.cpu().numpy()

    #print(n_age)
    #print(n_age_pred)
    #print(n_sex)
    #print(n_sex_pred)

    error_age= n_age - n_age_pred
    rmse_age = np.sqrt(np.mean(np.square(error_age)))
    me = np.mean(error_age)
    mede = np.median(error_age)
    iqre = np.quantile(error_age, 0.75) - np.quantile(error_age, 0.25)
    stde = np.std(error_age)
    absolute_error_age = np.abs(error_age)
    mae = np.mean(absolute_error_age)
    medae = np.median(absolute_error_age)
    iqrae = np.quantile(absolute_error_age, 0.75) - np.quantile(absolute_error_age, 0.25)
    stdae = np.std(absolute_error_age)
    
    print("\nME: {:.2f}".format(me / 365), file=print_file)
    print("STDE: {:.2f}".format(stde / 365), file=print_file)
    print("MEDE: {:.2f}".format(mede / 365), file=print_file)
    print("IQRE: {:.2f}".format(iqre / 365), file=print_file)
    print("MAE: {:.2f}".format(mae / 365), file=print_file)
    print("MEDAE: {:.2f}".format(medae / 365), file=print_file)
    print("STDAE: {:.2f}".format(stdae / 365), file=print_file)
    print("IQRAE: {:.2f}".format(iqrae / 365), file=print_file)
    print("RMSE age: {:.2f}".format(rmse_age / 365), file=print_file)

    print("\nAccuracy sex: {:.2f}".format(accuracy_score(n_sex, np.round(n_sex_pred))),file=print_file)
    print("Sensitivity sex: {:.2f}".format(recall_score(n_sex, np.round(n_sex_pred))), file=print_file)
    print("Specificity sex: {:.2f}".format(precision_score(n_sex, np.round(n_sex_pred))), file=print_file)
    print("AUC sex: {:.2f}".format(roc_auc_score(n_sex, n_sex_pred)), file=print_file)

    return me

