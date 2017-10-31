import click


@click.command()
@click.option('--exp', type=click.Choice(['train_val', 'train_test',
                                          ]), default='train_val',
              help='experiment to run')
@click.option('--scale_u_range', type=str, default='',
              help='aug xform: uniform scale range lower:upper')
@click.option('--scale_x_range', type=str, default='',
              help='aug xform: scale x range lower:upper')
@click.option('--scale_y_range', type=str, default='',
              help='aug xform: scale y range lower:upper')
@click.option('--affine_std', type=float, default=0.0, help='aug xform: random affine transform std-dev')
@click.option('--xlat_range', type=float, default=0.0, help='aug xform: translation range')
@click.option('--rot_std', type=float, default=0.2, help='aug xform: rotation std-dev')
@click.option('--hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--colour_rot_std', type=float, default=0.05,
              help='aug colour; colour rotation standard deviation')
@click.option('--colour_off_std', type=float, default=0.1,
              help='aug colour; colour offset standard deviation')
@click.option('--greyscale', is_flag=True, default=True,
              help='aug colour; enable greyscale')
@click.option('--cutout_prob', type=float, default=0.0,
              help='aug cutout probability')
@click.option('--cutout_size', type=float, default=0.3,
              help='aug cutout size (fraction)')
@click.option('--batch_size', type=int, default=16, help='mini-batch size')
@click.option('--n_batches', type=int, default=1, help='number of batches to show')
@click.option('--seed', type=int, default=12345, help='random seed (0 for time-based)')
def experiment(exp, scale_u_range, scale_x_range, scale_y_range, affine_std, xlat_range, rot_std, hflip,
               intens_scale_range, colour_rot_std, colour_off_std, greyscale, cutout_prob, cutout_size,
               batch_size, n_batches, seed):
    import os
    import sys
    import cmdline_helpers
    intens_scale_range_lower, intens_scale_range_upper = cmdline_helpers.colon_separated_range(intens_scale_range)
    scale_u_range = cmdline_helpers.colon_separated_range(scale_u_range)
    scale_x_range = cmdline_helpers.colon_separated_range(scale_x_range)
    scale_y_range = cmdline_helpers.colon_separated_range(scale_y_range)


    import time
    import tqdm
    import math
    import numpy as np
    from matplotlib import pyplot as plt
    from batchup import data_source, work_pool
    import visda17_dataset
    import augmentation, image_transforms
    import itertools


    n_chn = 0

    mean_value = np.array([0.485, 0.456, 0.406])
    std_value = np.array([0.229, 0.224, 0.225])

    if exp == 'train_val':
        d_source = visda17_dataset.TrainDataset(img_size=(96, 96),
                                                mean_value=mean_value, std_value=std_value,
                                                range01=True, rgb_order=True,
                                                random_crop=False)
        d_target = visda17_dataset.ValidationDataset(img_size=(96, 96),
                                                     mean_value=mean_value, std_value=std_value,
                                                     range01=True, rgb_order=True,
                                                     random_crop=False)
        d_target_test = visda17_dataset.ValidationDataset(img_size=(96, 96),
                                                     mean_value=mean_value, std_value=std_value,
                                                     range01=True, rgb_order=True,
                                                     random_crop=False)
    elif exp == 'train_test':
        print('train_test experiment not supported yet')
        return
    else:
        print('Unknown experiment type \'{}\''.format(exp))
        return

    n_classes = d_source.n_classes
    n_domains = 2

    print('Loaded data')


    arch = 'show-images'


    # Image augmentation

    aug = augmentation.ImageAugmentation(
        hflip, xlat_range, affine_std, rot_std=rot_std,
        intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
        colour_rot_std=colour_rot_std, colour_off_std=colour_off_std, greyscale=greyscale,
        scale_u_range=scale_u_range, scale_x_range=scale_x_range, scale_y_range=scale_y_range,
        cutout_size=cutout_size, cutout_probability=cutout_prob)

    # Report setttings
    print('sys.argv={}'.format(sys.argv))

    # Report dataset size
    print('Dataset:')
    print('SOURCE len(X)={}, y.shape={}'.format(len(d_source.images), d_source.y.shape))
    print('TARGET len(X)={}'.format(len(d_target.images)))

    print('Building data sources...')
    source_train_ds = data_source.ArrayDataSource([d_source.images, d_source.y], repeats=-1)
    target_train_ds = data_source.ArrayDataSource([d_target.images], repeats=-1)
    train_ds = data_source.CompositeDataSource([source_train_ds, target_train_ds])

    border_value = int(np.mean(mean_value) * 255 + 0.5)

    train_xf = image_transforms.Compose(
        image_transforms.ScaleCropAndAugmentAffine((96, 96), (16, 16), True, aug, border_value, mean_value, std_value),
        image_transforms.ToTensor(),
    )

    test_xf = image_transforms.Compose(
        image_transforms.ScaleAndCrop((96, 96), (16, 16), False),
        image_transforms.ToTensor(),
        image_transforms.Standardise(mean_value, std_value),
    )

    def augment(X_sup, y_sup, X_tgt):
        X_sup = train_xf(X_sup)[0]
        X_tgt_0 = train_xf(X_tgt)[0]
        X_tgt_1 = train_xf(X_tgt)[0]
        return [X_sup, y_sup, X_tgt_0, X_tgt_1]

    train_ds = train_ds.map(augment)

    test_ds = data_source.ArrayDataSource([d_target_test.images]).map(test_xf)

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    print('Showing...')

    n_shown = 0
    for (src_X, src_y, tgt_X0, tgt_X1), (te_X,) in zip(
            train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng),
            test_ds.batch_iterator(batch_size=batch_size)):
        print('Batch')
        tgt_X = np.zeros((tgt_X0.shape[0] + tgt_X1.shape[0],) + tgt_X0.shape[1:], dtype=np.float32)
        tgt_X[0::2] = tgt_X0
        tgt_X[1::2] = tgt_X1
        x = np.concatenate([src_X, tgt_X, te_X], axis=0)
        n = x.shape[0]
        n_sup = src_X.shape[0] + tgt_X.shape[0]
        across = int(math.ceil(math.sqrt(float(n))))
        plt.figure(figsize=(16,16))

        for i in tqdm.tqdm(range(n)):
            plt.subplot(across, across, i + 1)
            im_x = x[i] * std_value[:, None, None] + mean_value[:, None, None]
            im_x = np.clip(im_x, 0.0, 1.0)
            plt.imshow(im_x.transpose(1, 2, 0))
            if i < src_y.shape[0]:
                plt.title(str(src_y[i]))
            elif i < n_sup:
                plt.title('target')
            else:
                plt.title('test')
        plt.show()
        n_shown += 1
        if n_shown >= n_batches:
            break


if __name__ == '__main__':
    experiment()