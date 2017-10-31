import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


class ImageDataset (object):
    class ImageAccessor (object):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset.paths)

        def __getitem__(self, item):
            if isinstance(item, int):
                return self.dataset.load_image(self.dataset.paths[item])
            else:
                xs = []
                if isinstance(item, slice):
                    indices = range(*item.indices(len(self)))
                elif isinstance(item, np.ndarray):
                    indices = item
                else:
                    raise TypeError('item should be an int/long, a slice or an array, not a {}'.format(
                        type(item)
                    ))
                for i in indices:
                    img = self.dataset.load_image(self.dataset.paths[i])
                    xs.append(img)
                return xs


    def __init__(self, img_size, range01, rgb_order, class_names, n_classes, names, paths, y,
                 dummy=False):
        self.img_size = img_size

        self.range01 = range01
        self.rgb_order = rgb_order

        self.dummy = dummy

        self.images = self.ImageAccessor(self)

        self.class_names = class_names
        self.n_classes = n_classes
        self.names = names
        self.paths = paths
        if y is not None:
            self.y = np.array(y, dtype=np.int32)
            self.has_ground_truth = True
        else:
            self.has_ground_truth = False


    def load_image(self, path):
        if self.dummy:
            return np.random.randint(0, 256, size=self.img_size + (3,)).astype(np.uint8)
        else:
            img = cv2.imread(path)
            if self.rgb_order:
                img = img[:, :, ::-1]
            return img


    def prediction_evaluator(self, sample_indices=None):
        if not self.has_ground_truth:
            raise ValueError('Cannot create evaluator; dataset has no ground truth')

        if sample_indices is None:
            return PredictionEvaluator(self.y, self.n_classes, self.class_names)
        else:
            return PredictionEvaluator(self.y[sample_indices], self.n_classes, self.class_names)


class PredictionEvaluator (object):
    def __init__(self, y, n_classes, class_names):
        self.y = y
        self.n_classes = n_classes
        self.class_names = class_names
        self.hist = np.bincount(y, minlength=self.n_classes)


    def evaluate(self, tgt_pred_prob_y):
        tgt_pred_y = np.argmax(tgt_pred_prob_y, axis=1)
        aug_class_true_pos = np.zeros((self.n_classes,))

        # Compute per-class accuracy
        for cls_i in range(self.n_classes):
            aug_class_true_pos[cls_i] = ((self.y == cls_i) & (tgt_pred_y == cls_i)).sum()

        aug_cls_acc = aug_class_true_pos.astype(float) / np.maximum(self.hist.astype(float), 1.0)

        mean_aug_class_acc = aug_cls_acc.mean()

        aug_cls_acc_str = ',  '.join(['{}: {:.3%}'.format(self.class_names[cls_i], aug_cls_acc[cls_i])
                                      for cls_i in range(self.n_classes)])

        return mean_aug_class_acc, aug_cls_acc_str


def subset_indices(d_source, d_target, subsetsize, subsetseed):
    if subsetsize > 0:
        if subsetseed != 0:
            subset_rng = np.random.RandomState(subsetseed)
        else:
            subset_rng = np.random
        strat = StratifiedShuffleSplit(n_splits=1, test_size=subsetsize, random_state=subset_rng)
        shuf = ShuffleSplit(n_splits=1, test_size=subsetsize, random_state=subset_rng)
        _, source_indices = next(strat.split(d_source.y, d_source.y))
        n_src = source_indices.shape[0]
        if d_target.has_ground_truth:
            _, target_indices = next(strat.split(d_target.y, d_target.y))
        else:
            _, target_indices = next(shuf.split(np.arange(len(d_target.images))))
        n_tgt = target_indices.shape[0]
    else:
        source_indices = None
        target_indices = None
        n_src = len(d_source.images)
        n_tgt = len(d_target.images)

    return source_indices, target_indices, n_src, n_tgt
