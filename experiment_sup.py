import click


@click.command()
@click.option('--exp', type=click.Choice([
    'visda_train_val', 'visda_train_test',
    'office_amazon_dslr', 'office_amazon_webcam', 'office_dslr_amazon', 'office_dslr_webcam', 'office_webcam_amazon', 'office_webcam_dslr',
    ]), default='visda_train_val', help='experiment to run')
@click.option('--arch', type=click.Choice([
    '',
    'resnet50', 'resnet101', 'resnet152',
]), default='', help='network architecture')
@click.option('--rnd_init', is_flag=True, default=False)
@click.option('--img_size', type=int, default=96, help='input image size')
@click.option('--standardise_samples', is_flag=True, default=False)
@click.option('--learning_rate', type=float, default=1e-4, help='learning rate (Adam)')
@click.option('--pretrained_lr_factor', type=float, default=0.1,
              help='learning rate scale factor for pre-trained layers')
@click.option('--fix_layers', type=str, default='',
              help='List of layers to fix')
@click.option('--double_softmax', is_flag=True, default=False)
@click.option('--use_dropout', is_flag=True, default=False)
@click.option('--scale_u_range', type=str, default='',
              help='aug xform: scale uniform range; lower:upper')
@click.option('--scale_x_range', type=str, default='',
              help='aug xform: scale x range; lower:upper')
@click.option('--scale_y_range', type=str, default='',
              help='aug xform: scale y range; lower:upper')
@click.option('--affine_std', type=float, default=0.1, help='aug xform: random affine transform std-dev')
@click.option('--xlat_range', type=float, default=0.0, help='aug xform: translation range')
@click.option('--rot_std', type=float, default=0.2, help='aug xform: rotation std-dev')
@click.option('--hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--colour_rot_std', type=float, default=0.05,
              help='aug colour; colour rotation standard deviation')
@click.option('--colour_off_std', type=float, default=0.1,
              help='aug colour; colour offset standard deviation')
@click.option('--greyscale', is_flag=True, default=False,
              help='aug colour; enable greyscale')
@click.option('--img_pad_width', type=int, default=0,
              help='random cropping; padding width')
@click.option('--num_epochs', type=int, default=200, help='number of epochs')
@click.option('--batch_size', type=int, default=64, help='mini-batch size')
@click.option('--seed', type=int, default=0, help='random seed (0 for time-based)')
@click.option('--log_file', type=str, default='', help='log file path (none to disable)')
@click.option('--result_file', type=str, default='',
              help='path to HFD5 file to save results to')
@click.option('--hide_progress_bar', is_flag=True, default=False, help='Hide training progress bar')
@click.option('--subsetsize', type=int, default=0, help='Subset size (0 to use full dataset)')
@click.option('--subsetseed', type=int, default=0, help='subset random seed (0 for time based)')
@click.option('--device', type=int, default=0, help='Device')
def experiment(exp, arch, rnd_init, img_size, standardise_samples,
               learning_rate, pretrained_lr_factor, fix_layers,
               double_softmax, use_dropout,
               scale_u_range, scale_x_range, scale_y_range,
               affine_std, xlat_range, rot_std, hflip,
               intens_scale_range, colour_rot_std, colour_off_std, greyscale, img_pad_width,
               num_epochs, batch_size, seed,
               log_file, result_file, hide_progress_bar,
               subsetsize, subsetseed,
               device):
    settings = locals().copy()

    if rnd_init:
        if fix_layers != '':
            print('`rnd_init` and `fix_layers` are mutually exclusive')
            return

    import os
    import sys
    import cmdline_helpers

    fix_layers = [lyr.strip() for lyr in fix_layers.split(',')]

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None

    if log_file is not None:
        if os.path.exists(log_file):
            print('Output log file {} already exists'.format(log_file))
            return

    intens_scale_range_lower, intens_scale_range_upper = cmdline_helpers.colon_separated_range(intens_scale_range)
    scale_u_range = cmdline_helpers.colon_separated_range(scale_u_range)
    scale_x_range = cmdline_helpers.colon_separated_range(scale_x_range)
    scale_y_range = cmdline_helpers.colon_separated_range(scale_y_range)


    import time
    import tqdm
    import math
    import tables
    import numpy as np
    from batchup import data_source, work_pool
    import image_dataset, visda17_dataset, office_dataset
    import network_architectures
    import augmentation
    import image_transforms
    from sklearn.model_selection import StratifiedShuffleSplit
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F

    if hide_progress_bar:
        progress_bar = None
    else:
        progress_bar = tqdm.tqdm


    with torch.cuda.device(device):
        pool = work_pool.WorkerThreadPool(4)

        n_chn = 0
        half_batch_size = batch_size // 2

        RESNET_ARCHS = {'resnet50', 'resnet101', 'resnet152'}
        RNDINIT_ARCHS = {'vgg13_48_gp'}

        if arch == '':
            if exp in {'train_val', 'train_test'}:
                arch = 'resnet50'

        if arch in RESNET_ARCHS and not rnd_init:
            mean_value = np.array([0.485, 0.456, 0.406])
            std_value = np.array([0.229, 0.224, 0.225])
        elif arch in RNDINIT_ARCHS:
            mean_value = np.array([0.5, 0.5, 0.5])
            std_value = np.array([0.5, 0.5, 0.5])
            rnd_init = True
        else:
            mean_value = std_value = None


        img_shape = (img_size, img_size)
        img_padding = (img_pad_width, img_pad_width)

        if exp == 'visda_train_val':
            d_source = visda17_dataset.TrainDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = visda17_dataset.ValidationDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'visda_train_test':
            d_source = visda17_dataset.TrainDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = visda17_dataset.TestDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_amazon_dslr':
            d_source = office_dataset.OfficeAmazonDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeDSLRDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_amazon_webcam':
            d_source = office_dataset.OfficeAmazonDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeWebcamDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_dslr_amazon':
            d_source = office_dataset.OfficeDSLRDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeAmazonDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_dslr_webcam':
            d_source = office_dataset.OfficeDSLRDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeWebcamDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_webcam_amazon':
            d_source = office_dataset.OfficeWebcamDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeAmazonDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'office_webcam_dslr':
            d_source = office_dataset.OfficeWebcamDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = office_dataset.OfficeDSLRDataset(img_size=img_shape, range01=True, rgb_order=True)
        else:
            print('Unknown experiment type \'{}\''.format(exp))
            return

        #
        # Result file
        #

        if result_file != '':
            cmdline_helpers.ensure_containing_dir_exists(result_file)
            h5_filters = tables.Filters(complevel=9, complib='blosc')
            f_target_pred = tables.open_file(result_file, mode='w')
            g_tgt_pred = f_target_pred.create_group(f_target_pred.root, 'target_pred_y', 'Target prediction')
            arr_tgt_pred = f_target_pred.create_earray(g_tgt_pred, 'y', tables.Float32Atom(),
                                                       (0, len(d_target.images), d_target.n_classes),
                                                       filters=h5_filters)
        else:
            f_target_pred = None
            g_tgt_pred = None
            arr_tgt_pred = None


        # Delete the training ground truths as we should not be using them
        # del d_target.y

        n_classes = d_source.n_classes

        print('Loaded data')

        net_class = network_architectures.get_build_fn_for_architecture(arch)

        net = net_class(n_classes, img_size, use_dropout, not rnd_init).cuda()

        if arch in RESNET_ARCHS and not rnd_init:
            named_params = list(net.named_parameters())
            new_params = []
            pretrained_params = []
            for name, param in named_params:
                if name.startswith('new_'):
                    new_params.append(param)
                else:
                    fix = False
                    for lyr in fix_layers:
                        if name.startswith(lyr + '.'):
                            fix = True
                            break
                    if not fix:
                        pretrained_params.append(param)
                    else:
                        print('Fixing param {}'.format(name))
                        param.requires_grad = False

            new_optimizer = torch.optim.Adam(new_params, lr=learning_rate)
            if len(pretrained_params) > 0:
                pretrained_optimizer = torch.optim.Adam(pretrained_params, lr=learning_rate * pretrained_lr_factor)
            else:
                pretrained_optimizer = None
        else:
            new_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            pretrained_optimizer = None
        classification_criterion = nn.CrossEntropyLoss()

        print('Built network')


        # Image augmentation

        aug = augmentation.ImageAugmentation(
            hflip, xlat_range, affine_std, rot_std=rot_std,
            intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
            colour_rot_std=colour_rot_std, colour_off_std=colour_off_std, greyscale=greyscale,
            scale_u_range=scale_u_range, scale_x_range=scale_x_range, scale_y_range=scale_y_range)

        test_aug = augmentation.ImageAugmentation(
            hflip, xlat_range, 0.0, rot_std=0.0,
            scale_u_range=scale_u_range, scale_x_range=scale_x_range, scale_y_range=scale_y_range)

        border_value = int(np.mean(mean_value) * 255 + 0.5)

        sup_xf = image_transforms.Compose(
            image_transforms.ScaleCropAndAugmentAffine(img_shape, img_padding, True, aug, border_value, mean_value,
                                                       std_value),
            image_transforms.ToTensor(),
        )

        test_xf = image_transforms.Compose(
            image_transforms.ScaleAndCrop(img_shape, img_padding, False),
            image_transforms.ToTensor(),
            image_transforms.Standardise(mean_value, std_value),
        )

        test_xf_aug_mult = image_transforms.Compose(
            image_transforms.ScaleCropAndAugmentAffineMultiple(
                16, img_shape, img_padding, True, test_aug, border_value, mean_value, std_value),
            image_transforms.ToTensorMultiple(),
        )


        def augment(X_sup, y_sup):
            X_sup = sup_xf(X_sup)[0]
            return X_sup, y_sup


        _one = torch.autograd.Variable(torch.from_numpy(np.array([1.0]).astype(np.float32)).cuda())
        def f_train(X_sup, y_sup):
            X_sup = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            y_sup = torch.autograd.Variable(torch.from_numpy(y_sup).long().cuda())

            if pretrained_optimizer is not None:
                pretrained_optimizer.zero_grad()
            new_optimizer.zero_grad()
            net.train(mode=True)

            sup_logits_out = net(X_sup)

            # Supervised classification loss
            if double_softmax:
                clf_loss = classification_criterion(F.softmax(sup_logits_out), y_sup)
            else:
                clf_loss = classification_criterion(sup_logits_out, y_sup)

            loss_expr = clf_loss

            loss_expr.backward()
            if pretrained_optimizer is not None:
                pretrained_optimizer.step()
            new_optimizer.step()

            n_samples = X_sup.size()[0]

            return (float(clf_loss.data.cpu()[0]) * n_samples,)

        print('Compiled training function')

        def f_pred(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            net.train(mode=False)
            return F.softmax(net(X_var)).data.cpu().numpy()

        def f_pred_tgt_mult(X_sup):
            net.train(mode=False)
            y_pred_aug = []
            for aug_i in range(len(X_sup)):
                X_var = torch.autograd.Variable(torch.from_numpy(X_sup[aug_i, ...]).cuda())
                y_pred = F.softmax(net(X_var)).data.cpu().numpy()
                y_pred_aug.append(y_pred[None, ...])
            y_pred_aug = np.concatenate(y_pred_aug, axis=0)
            return (y_pred_aug.mean(axis=0),)

        print('Compiled evaluation function')

        # Setup output
        def log(text):
            print(text)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(text + '\n')
                    f.flush()
                    f.close()

        cmdline_helpers.ensure_containing_dir_exists(log_file)

        # Report setttings
        log('Program = {}'.format(sys.argv[0]))
        log('Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

        # Report dataset size
        log('Dataset:')
        print('SOURCE len(X)={}, y.shape={}'.format(len(d_source.images), d_source.y.shape))
        print('TARGET len(X)={}'.format(len(d_target.images)))


        # Subset
        source_indices, target_indices, n_src, n_tgt = image_dataset.subset_indices(
            d_source, d_target, subsetsize, subsetseed
        )


        n_train_batches = n_src // batch_size + 1
        n_test_batches = n_tgt // (batch_size * 2) + 1

        print('Training...')
        train_ds = data_source.ArrayDataSource([d_source.images, d_source.y], indices=source_indices)
        train_ds = train_ds.map(augment)
        train_ds = pool.parallel_data_source(train_ds, batch_buffer_size=min(20, n_train_batches))

        # source_test_ds = data_source.ArrayDataSource([d_source.images])
        # source_test_ds = pool.parallel_data_source(source_test_ds)
        target_ds_for_test = data_source.ArrayDataSource([d_target.images], indices=target_indices)
        target_test_ds = target_ds_for_test.map(test_xf)
        target_test_ds = pool.parallel_data_source(target_test_ds, batch_buffer_size=min(20, n_test_batches))
        target_mult_test_ds = target_ds_for_test.map(test_xf_aug_mult)
        target_mult_test_ds = pool.parallel_data_source(target_mult_test_ds, batch_buffer_size=min(20, n_test_batches))


        if seed != 0:
            shuffle_rng = np.random.RandomState(seed)
        else:
            shuffle_rng = np.random


        if d_target.has_ground_truth:
            evaluator = d_target.prediction_evaluator(target_indices)
        else:
            evaluator = None


        train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

        for epoch in range(num_epochs):
            t1 = time.time()

            test_batch_iter = target_test_ds.batch_iterator(batch_size=batch_size)

            train_clf_loss, = data_source.batch_map_mean(
                f_train, train_batch_iter, n_batches=n_train_batches,
                progress_iter_func=progress_bar)
            # train_clf_loss, train_unsup_loss, mask_rate, train_align_loss = train_ds.batch_map_mean(
            #     lambda *x: 1.0, batch_size=batch_size, shuffle=shuffle_rng, n_batches=n_train_batches,
            #     progress_iter_func=progress_bar)


            if d_target.has_ground_truth or arr_tgt_pred is not None:
                tgt_pred_prob_y, = data_source.batch_map_concat(f_pred, test_batch_iter,
                                                                progress_iter_func=progress_bar)
            else:
                tgt_pred_prob_y = None

            train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

            if d_target.has_ground_truth:
                mean_class_acc, cls_acc_str = evaluator.evaluate(tgt_pred_prob_y)

                t2 = time.time()

                log('Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
                    'TGT mean class acc={:.3%}'.format(
                    epoch, t2 - t1, train_clf_loss, mean_class_acc))
                log('  per class:  {}'.format(cls_acc_str))
            else:
                t2 = time.time()

                log('Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}'.format(
                    epoch, t2 - t1, train_clf_loss))

            # Save results
            if arr_tgt_pred is not None:
                arr_tgt_pred.append(tgt_pred_prob_y[None, ...].astype(np.float32))



        # Predict on test set, using augmentation
        tgt_aug_pred_prob_y, = target_mult_test_ds.batch_map_concat(f_pred_tgt_mult, batch_size=batch_size,
                                                           progress_iter_func=progress_bar)
        if d_target.has_ground_truth:
            aug_mean_class_acc, aug_cls_acc_str = evaluator.evaluate(tgt_aug_pred_prob_y)

            log('FINAL: TGT AUG mean class acc={:.3%}'.format(aug_mean_class_acc))
            log('  per class:  {}'.format(aug_cls_acc_str))

        if f_target_pred is not None:
            f_target_pred.create_array(g_tgt_pred, 'y_prob', tgt_pred_prob_y)
            f_target_pred.create_array(g_tgt_pred, 'y_prob_aug', tgt_aug_pred_prob_y)
            f_target_pred.close()

if __name__ == '__main__':
    experiment()