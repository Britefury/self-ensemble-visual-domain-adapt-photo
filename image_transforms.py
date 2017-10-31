import numpy as np
import cv2
import augmentation
from skimage.util import img_as_float


def _compute_scale_and_crop(image_size, crop_size, padding, random_crop):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    scale_y = float(image_size[0]) / float(padded_size[0])
    scale_x = float(image_size[1]) / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factor = min(scale_y, scale_x)

    # Compute the size of the region that we must extract from the image
    region_height = int(float(crop_size[0]) * scale_factor + 0.5)
    region_width = int(float(crop_size[1]) * scale_factor + 0.5)

    # Compute the additional space available
    if scale_x > scale_y:
        # Crop in X
        extra_x = image_size[1] - region_width
        extra_y = padding[0]
    else:
        # Crop in Y
        extra_y = image_size[0] - region_height
        extra_x = padding[1]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        pos_y = np.random.randint(0, extra_y + 1, size=(1,))[0]
        pos_x = np.random.randint(0, extra_x + 1, size=(1,))[0]
    else:
        pos_y = extra_y // 2
        pos_x = extra_x // 2

    return (pos_y, pos_x), (region_height, region_width)



def _compute_scales_and_crops(image_sizes, crop_size, padding, random_crop):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    image_sizes = image_sizes.astype(float)
    scale_ys = image_sizes[:, 0] / float(padded_size[0])
    scale_xs = image_sizes[:, 1] / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factors = np.minimum(scale_ys, scale_xs)

    # Compute the size of the region that we must extract from the image
    region_sizes = (np.array(crop_size)[None, :] * scale_factors[:, None] + 0.5).astype(int)

    # Compute the additional space available
    extra_space = np.repeat(np.array(padding, dtype=int)[None, :], image_sizes.shape[0], axis=0)
    # Crop in X
    crop_in_x = scale_xs > scale_ys
    extra_space[crop_in_x, 1] = image_sizes[crop_in_x, 1] - region_sizes[crop_in_x, 1]
    # Crop in Y
    crop_in_y = ~crop_in_x
    extra_space[crop_in_y, 0] = image_sizes[crop_in_y, 0] - region_sizes[crop_in_y, 0]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        t = np.random.uniform(0.0, 1.0, size=image_sizes.shape)
        pos = (t * (extra_space + 1.0)).astype(int)
    else:
        pos = extra_space // 2

    return pos, region_sizes


def _compute_scales_and_crops_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size):
    padded_size = crop_size[0] + padding[0], crop_size[1] + padding[1]
    # Compute size ratio from the padded region size to the image_size
    image_sizes = image_sizes.astype(float)
    scale_ys = image_sizes[:, 0] / float(padded_size[0])
    scale_xs = image_sizes[:, 1] / float(padded_size[1])
    # Take the minimum as this is the factor by which we must scale to take a `padded_size` sized chunk
    scale_factors = np.minimum(scale_ys, scale_xs)

    # Compute the size of the region that we must extract from the image
    region_sizes = (np.array(crop_size)[None, :] * scale_factors[:, None] + 0.5).astype(int)

    # Compute the additional space available
    extra_space = np.repeat(np.array(padding, dtype=int)[None, :], image_sizes.shape[0], axis=0)
    # Crop in X
    crop_in_x = scale_xs > scale_ys
    extra_space[crop_in_x, 1] = image_sizes[crop_in_x, 1] - region_sizes[crop_in_x, 1]
    # Crop in Y
    crop_in_y = ~crop_in_x
    extra_space[crop_in_y, 0] = image_sizes[crop_in_y, 0] - region_sizes[crop_in_y, 0]

    # Either choose the centre piece or choose a random piece
    if random_crop:
        t = np.random.uniform(0.0, 1.0, size=image_sizes.shape)
        pos_a = (t * (extra_space + 1.0)).astype(int)
        if pair_offset_size > 0:
            pos_b_off = np.random.randint(-pair_offset_size, pair_offset_size, size=image_sizes.shape)
            pos_b = np.clip(pos_a + pos_b_off, 0, extra_space)
        else:
            pos_b = pos_a
    else:
        pos_a = extra_space // 2
        pos_b = pos_a

    return pos_a, pos_b, region_sizes



def _compute_scale_and_crop_matrix(img_size, crop_size, padding, random_crop):
    (pos_y, pos_x), (reg_h, reg_w) = _compute_scale_and_crop(img_size, crop_size, padding, random_crop)

    scale_y = float(crop_size[0]) / float(reg_h)
    scale_x = float(crop_size[1]) / float(reg_w)
    off_y = float(pos_y) * scale_y
    off_x = float(pos_x) * scale_x

    scale_and_crop_matrix = np.array([
        [scale_x, 0.0, -off_x,],
        [0.0, scale_y, -off_y]
    ])

    return scale_and_crop_matrix


def _positions_and_sizes_to_matrices(pos, sz, crop_size):
    scales = np.array(crop_size, dtype=float)[None, :] / sz
    offsets = pos * scales

    scale_and_crop_matrices = np.zeros((scales.shape[0], 2, 3))
    scale_and_crop_matrices[:, 0, 0] = scales[:, 1]
    scale_and_crop_matrices[:, 1, 1] = scales[:, 0]
    scale_and_crop_matrices[:, :, 2] = -offsets[:,::-1]

    return scale_and_crop_matrices


def _compute_scale_and_crop_matrices(image_sizes, crop_size, padding, random_crop):
    pos, sz = _compute_scales_and_crops(image_sizes, crop_size, padding, random_crop)
    return _positions_and_sizes_to_matrices(pos, sz, crop_size)


def _compute_scale_and_crop_matrix_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size):
    pos_a, pos_b, sz = _compute_scales_and_crops_pairs(image_sizes, crop_size, padding, random_crop, pair_offset_size)

    mtx_a = _positions_and_sizes_to_matrices(pos_a, sz, crop_size)
    mtx_b = _positions_and_sizes_to_matrices(pos_b, sz, crop_size)
    return mtx_a, mtx_b



class ImageTransform (object):
    def __call__(self, batch):
        raise NotImplementedError('Abstract for {}'.format(type(self)))


class Compose(ImageTransform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)[0]
        return (images,)


class ScaleAndCrop (ImageTransform):
    def __init__(self, crop_size, padding, random_crop):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop


    def __call__(self, images):
        image_sizes = np.array([list(img.shape[:2]) for img in images])

        pos, sz = _compute_scales_and_crops(image_sizes, self.crop_size, self.padding, self.random_crop)

        result = []
        for i, img in enumerate(images):
            # Compute scale factor to maintain aspect ratio
            cropped_img = img[pos[i,0]:pos[i,0] + sz[i,0], pos[i,1]:pos[i,1] + sz[i,1], :]

            # Scale
            result.append(cv2.resize(cropped_img, (self.crop_size[1], self.crop_size[0])))

        return (result,)


class ScaleAndCropAffine (ImageTransform):
    def __init__(self, crop_size, padding, random_crop):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop


    def __call__(self, images):
        image_sizes = np.array([list(img.shape[:2]) for img in images])
        scale_and_crop_matrices = _compute_scale_and_crop_matrices(image_sizes, self.crop_size, self.padding, self.random_crop)

        result = []
        for i, img in enumerate(images):
            img = cv2.warpAffine(img, scale_and_crop_matrices[i, :, :], self.crop_size[::-1])
            result.append(img)

        return (result,)


class ScaleCropAndAugmentAffine (ImageTransform):
    def __init__(self, crop_size, padding, random_crop, aug, border_value, mean_value, std_value):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop
        self.aug = aug
        self.border_value = border_value
        self.mean = mean_value
        self.std = std_value

    def __call__(self, images):
        image_sizes = np.array([list(img.shape[:2]) for img in images])
        scale_crop_mtx = _compute_scale_and_crop_matrices(image_sizes, self.crop_size, self.padding, self.random_crop)

        aug_xf = self.aug.aug_xforms(len(images), self.crop_size)
        scale_crop_aug_xf = augmentation.cat_nx2x3(aug_xf, scale_crop_mtx)
        colour_matrix, colour_offset = self.aug.aug_colour_xforms(len(images))
        cutout_flags, cutout_lower, cutout_upper = self.aug.aug_cutouts(len(images), self.crop_size)

        result = []
        for i, img in enumerate(images):
            img = cv2.warpAffine(img, scale_crop_aug_xf[i, :, :], self.crop_size[::-1], borderValue=self.border_value,
                                 borderMode=cv2.BORDER_REFLECT_101)
            img = img_as_float(img).astype(np.float32)
            img = (img - self.mean[None, None, :]) / self.std[None, None, :]
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
            if cutout_flags is not None and cutout_flags[i]:
                img[cutout_lower[i, 0]:cutout_upper[i, 0], cutout_lower[i, 1]:cutout_upper[i, 1], :] = 0.0
            result.append(img)

        return (result,)


class ScaleCropAndAugmentAffinePair (ImageTransform):
    def __init__(self, crop_size, padding, pair_offset_size, random_crop, aug, border_value, mean_value, std_value):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.crop_size = crop_size
        self.padding = padding
        self.pair_offset_size = pair_offset_size
        self.random_crop = random_crop
        self.aug = aug
        self.border_value = border_value
        self.mean = mean_value
        self.std = std_value

    def __call__(self, images):
        image_sizes = np.array([list(img.shape[:2]) for img in images])
        scale_crop_mtx_a, sclcrop_mtx_b = _compute_scale_and_crop_matrix_pairs(
            image_sizes, self.crop_size, self.padding, self.random_crop, self.pair_offset_size)

        aug_xf_a = self.aug.aug_xforms(len(images), self.crop_size)
        aug_xf_b = self.aug.aug_xforms(len(images), self.crop_size)
        scale_crop_aug_xf_a = augmentation.cat_nx2x3(aug_xf_a, scale_crop_mtx_a)
        scale_crop_aug_xf_b = augmentation.cat_nx2x3(aug_xf_b, sclcrop_mtx_b)
        colour_matrix_a, colour_offset_a = self.aug.aug_colour_xforms(len(images))
        colour_matrix_b, colour_offset_b = self.aug.aug_colour_xforms(len(images))
        cutout_flags_a, cutout_lower_a, cutout_upper_a = self.aug.aug_cutouts(len(images), self.crop_size)
        cutout_flags_b, cutout_lower_b, cutout_upper_b = self.aug.aug_cutouts(len(images), self.crop_size)

        result_a = []
        result_b = []
        for i, img in enumerate(images):
            img_a = cv2.warpAffine(img, scale_crop_aug_xf_a[i, :, :], self.crop_size[::-1],
                                  borderValue=self.border_value,
                                   borderMode=cv2.BORDER_REFLECT_101)
            img_b = cv2.warpAffine(img, scale_crop_aug_xf_b[i, :, :], self.crop_size[::-1],
                                  borderValue=self.border_value,
                                   borderMode=cv2.BORDER_REFLECT_101)
            img_a = img_as_float(img_a).astype(np.float32)
            img_b = img_as_float(img_b).astype(np.float32)
            img_a = (img_a - self.mean[None, None, :]) / self.std[None, None, :]
            img_b = (img_b - self.mean[None, None, :]) / self.std[None, None, :]
            img_a = np.tensordot(img_a, colour_matrix_a[i, :, :], [[2], [1]]) + colour_offset_a[i, None, None, :]
            img_b = np.tensordot(img_b, colour_matrix_b[i, :, :], [[2], [1]]) + colour_offset_b[i, None, None, :]
            if cutout_flags_a is not None and cutout_flags_a[i]:
                img_a[cutout_lower_a[i, 0]:cutout_upper_a[i, 0], cutout_lower_a[i, 1]:cutout_upper_a[i, 1], :] = 0.0
            if cutout_flags_b is not None and cutout_flags_b[i]:
                img_b[cutout_lower_b[i, 0]:cutout_upper_b[i, 0], cutout_lower_b[i, 1]:cutout_upper_b[i, 1], :] = 0.0
            result_a.append(img_a)
            result_b.append(img_b)

        return (result_a + result_b,)


class ScaleCropAndAugmentAffineMultiple (ImageTransform):
    def __init__(self, N, crop_size, padding, random_crop, aug, border_value, mean_value, std_value):
        """
        Constructor

        Section `(height + pad_y, width + pad_x)` will be taken from the centre
        of the image
        A `(height, width)` section will be taken from the centre of that

        :param crop_size: the desired image size `(height, width)`
        :param padding: padding around the crop `(pad_y, pad_x)`
        """
        self.N = N
        self.crop_size = crop_size
        self.padding = padding
        self.random_crop = random_crop
        self.aug = aug
        self.border_value = border_value
        self.mean = mean_value
        self.std = std_value

    def __call__(self, images):
        image_sizes = np.array([list(img.shape[:2]) for img in images])

        result = []
        for aug_i in range(self.N):
            scale_crop_mtx = _compute_scale_and_crop_matrices(
                image_sizes, self.crop_size, self.padding, self.random_crop)

            aug_xf = self.aug.aug_xforms(len(images), self.crop_size)
            scale_crop_aug_xf = augmentation.cat_nx2x3(aug_xf, scale_crop_mtx)
            colour_matrix, colour_offset = self.aug.aug_colour_xforms(len(images))
            cutout_flags, cutout_lower, cutout_upper = self.aug.aug_cutouts(len(images), self.crop_size)

            result_aug = []
            for i, img in enumerate(images):
                img = cv2.warpAffine(img, scale_crop_aug_xf[i, :, :], self.crop_size[::-1],
                                      borderValue=self.border_value,
                                       borderMode=cv2.BORDER_REFLECT_101)
                img = img_as_float(img).astype(np.float32)
                img = (img - self.mean[None, None, :]) / self.std[None, None, :]
                img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
                if cutout_flags is not None and cutout_flags[i]:
                    img[cutout_lower[i, 0]:cutout_upper[i, 0], cutout_lower[i, 1]:cutout_upper[i, 1], :] = 0.0
                result_aug.append(img)

            result.append(result_aug)

        return (result,)



class ToTensor (ImageTransform):
    def __call__(self, images):
        xs = []
        for img in images:
            img = img_as_float(img).astype(np.float32)
            img = img.transpose(2, 0, 1)[None, ...]
            xs.append(img)
        return (np.concatenate(xs, axis=0).astype(np.float32),)


class ToTensorMultiple (ImageTransform):
    def __call__(self, groups_of_images):
        res = []
        for group in groups_of_images:
            xs = []
            for img in group:
                img = img_as_float(img).astype(np.float32)
                img = img.transpose(2, 0, 1)[None, ...]
                xs.append(img)
            xs = np.concatenate(xs, axis=0).astype(np.float32)
            res.append(xs[None, ...])
        return (np.concatenate(res, axis=0).astype(np.float32),)


class Standardise (ImageTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        if isinstance(images, list):
            result = []
            for img in images:
                dtype = img.dtype
                img = (img - self.mean[None, None, :]) / self.std[None, None, :]
                result.append(img.astype(dtype))
            return (result,)
        else:
            dtype = images.dtype
            images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
            return (images.astype(dtype),)
