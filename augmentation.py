import numpy as np
import cv2


def join_xf_col(xf, col_mtx, col_off):
    return np.concatenate([xf.reshape((-1, 6)),
                           col_mtx.reshape((-1, 9)),
                           col_off.reshape((-1, 3))], axis=1).astype(np.float32)


def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def rotation_matrices(thetas):
    """
    Generate rotation matrices
    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    N = thetas.shape[0]
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = np.cos(thetas)
    rot_xf[:, 1, 0] = np.sin(thetas)
    rot_xf[:, 0, 1] = -np.sin(thetas)
    return rot_xf

def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


def identity_col_mtx(N):
    """
    Construct N identity 3x3 colour matrices
    :return: array of shape (N, 3, 3)
    """
    col_mtx = np.zeros((N, 3, 3), dtype=np.float32)
    col_mtx[:, 0, 0] = col_mtx[:, 1, 1] = col_mtx[:, 2, 2] = 1.0
    return col_mtx


def identity_col_off(N):
    """
    Construct N identity colour offsets
    :return: array of shape (N, 3)
    """
    col_off = np.zeros((N, 3), dtype=np.float32)
    return col_off


def identity_xf_col(N):
    xf = identity_xf(N)
    col_mtx = identity_col_mtx(N)
    col_off = identity_col_off(N)
    return join_xf_col(xf, col_mtx, col_off)


def axis_angle_rotation_matrices(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # Normalize
    axis = axis/np.sqrt(np.sum(axis*axis, axis=1, keepdims=True))
    a = np.cos(theta/2)
    axis_sin_theta = -axis*np.sin(theta/2)[:, None]
    b = axis_sin_theta[:, 0]
    c = axis_sin_theta[:, 1]
    d = axis_sin_theta[:, 2]
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot.transpose(2, 0, 1)




class ImageAugmentation (object):
    def __init__(self, hflip, xlat_range, affine_std, rot_std=0.0,
                 intens_scale_range_lower=None, intens_scale_range_upper=None,
                 colour_rot_std=0.0, colour_off_std=0.0,
                 greyscale=False,
                 scale_u_range=None, scale_x_range=None, scale_y_range=None,
                 cutout_size=None, cutout_probability=0.0):
        self.hflip = hflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.rot_std = rot_std
        self.intens_scale_range_lower = intens_scale_range_lower
        self.intens_scale_range_upper = intens_scale_range_upper
        self.colour_rot_std = colour_rot_std
        self.colour_off_std = colour_off_std
        self.greyscale = greyscale
        self.scale_u_range = scale_u_range
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.cutout_size = cutout_size
        self.cutout_probability = cutout_probability


    def aug_xforms(self, N, image_size):
        xf = identity_xf(N)

        if self.hflip:
            x_hflip = np.random.binomial(1, 0.5, size=(N,)) * 2 - 1
            xf[:, 0, 0] = x_hflip.astype(np.float32)

        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scl = np.exp(np.random.uniform(low=np.log(self.scale_u_range[0]), high=np.log(self.scale_u_range[1]), size=(N,)))
            xf[:, 0, 0] *= scl
            xf[:, 1, 1] *= scl
        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= np.exp(np.random.uniform(low=np.log(self.scale_x_range[0]), high=np.log(self.scale_x_range[1]), size=(N,)))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= np.exp(np.random.uniform(low=np.log(self.scale_y_range[0]), high=np.log(self.scale_y_range[1]), size=(N,)))

        if self.affine_std > 0.0:
            xf[:, :, :2] += np.random.normal(scale=self.affine_std, size=(N, 2, 2))
        xlat_y_bounds = self.xlat_range * 2.0 / float(image_size[0])
        xlat_x_bounds = self.xlat_range * 2.0 / float(image_size[1])
        xf[:, 0, 2] += np.random.uniform(low=-xlat_x_bounds, high=xlat_x_bounds, size=(N,))
        xf[:, 1, 2] += np.random.uniform(low=-xlat_y_bounds, high=xlat_y_bounds, size=(N,))

        if self.rot_std > 0.0:
            thetas = np.random.normal(scale=self.rot_std, size=(N,))
            rot_xf = rotation_matrices(thetas)
            xf = cat_nx2x3(xf, rot_xf)

        return centre_xf(xf, image_size)

    def aug_colour_xforms(self, N):
        colour_matrix = np.zeros((N, 3, 3))
        colour_matrix[:, 0, 0] = colour_matrix[:, 1, 1] = colour_matrix[:, 2, 2] = 1.0
        if self.colour_rot_std > 0.0:
            # Colour rotation: random thetas
            col_rot_thetas = np.random.normal(scale=self.colour_rot_std, size=(N,))
            # Colour rotation: random axes
            col_rot_axes = np.random.normal(size=(N, 3))
            invalid_axis_mask = np.dot(col_rot_axes, col_rot_axes.T) == 0

            # Re-draw invalid axes
            while invalid_axis_mask.any():
                col_rot_axes[col_rot_axes, :] = np.random.normal(scale=self.colour_rot_std,
                                                                 size=(int(invalid_axis_mask.sum()), 3))
                invalid_axis_mask = np.dot(col_rot_axes, col_rot_axes.T) == 0

            colour_matrix = axis_angle_rotation_matrices(col_rot_axes, col_rot_thetas)

        if self.greyscale:
            grey_factors = np.array([0.2125, 0.7154, 0.0721])
            grey_mtx = np.repeat(grey_factors[None, None, :], 3, axis=1)
            eye_mtx = np.eye(3)[None, :, :]
            factors = np.random.uniform(0.0, 1.0, size=(N, 1, 1))
            greyscale_mtx = eye_mtx + (grey_mtx - eye_mtx) * factors
            colour_matrix = np.matmul(colour_matrix, greyscale_mtx)

        colour_offset = np.zeros((N, 3))
        if self.colour_off_std > 0.0:
            colour_offset = np.random.normal(scale=self.colour_off_std, size=(N, 3))

        if self.intens_scale_range_lower is not None:
            col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                           size=(N,))
            colour_matrix = colour_matrix * col_factor[:, None, None]

        return colour_matrix, colour_offset


    def aug_colours(self, X):
        colour_matrix, colour_offset = self.aug_colour_xforms(X.shape[0])

        X_c = np.zeros_like(X)
        for i in range(X.shape[0]):
            img = X[i, :, :, :].transpose(1, 2, 0)
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]
            X_c[i, :, :, :] = img.transpose(2, 0, 1)

        return X_c


    def aug_cutouts(self, N, img_size):
        if self.cutout_probability > 0.0:
            img_size = np.array(img_size)
            cutout_shape = (np.array(img_size) * self.cutout_size + 0.5).astype(int)
            cutout_p = np.random.uniform(0.0, 1.0, size=(N,))
            cutout_pos_y = np.random.randint(0, img_size[0], size=(N,))
            cutout_pos_x = np.random.randint(0, img_size[1], size=(N,))
            cutout_pos = np.append(cutout_pos_y[:, None], cutout_pos_x[:, None], axis=1)
            cutout_lower = cutout_pos - (cutout_shape[None, :]//2)
            cutout_upper = cutout_lower + cutout_shape[None, :]

            cutout_flags = cutout_p <= self.cutout_probability
            cutout_lower = np.clip(cutout_lower, 0, img_size[None, :]-1)
            cutout_upper = np.clip(cutout_upper, 0, img_size[None, :]-1)

            return cutout_flags, cutout_lower, cutout_upper
        else:
            return None, None, None


    def augment(self, X):
        X = X.copy()
        N = X.shape[0]

        xf = self.aug_xforms(N, X.shape[2:])
        colour_matrix, colour_offset = self.aug_colour_xforms(X.shape[0])

        cutout_flags, cutout_lower, cutout_upper = self.aug_cutouts(N, X.shape[2:])

        for i in range(X.shape[0]):
            img = X[i, :, :, :].transpose(1, 2, 0)
            img = cv2.warpAffine(img, xf[i, :, :], (X.shape[3], X.shape[2]))
            img = np.tensordot(img, colour_matrix[i, :, :], [[2], [1]]) + colour_offset[i, None, None, :]

            if cutout_flags is not None and cutout_flags[i]:
                img[cutout_lower[i, 0]:cutout_upper[i,0], cutout_lower[i, 1]:cutout_upper[i,1], :] = 0.0

            X[i, :, :, :] = img.transpose(2, 0, 1)

        return X
