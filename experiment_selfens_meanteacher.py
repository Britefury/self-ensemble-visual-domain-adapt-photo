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
@click.option('--confidence_thresh', type=float, default=0.96, help='augmentation var loss confidence threshold')
@click.option('--teacher_alpha', type=float, default=0.99, help='Teacher EMA alpha (decay)')
@click.option('--unsup_weight', type=float, default=1.0, help='unsupervised loss weight')
@click.option('--cls_balance', type=float, default=0.001, help='Weight of class balancing component of unsupervised '
                                                              'loss')
@click.option('--cls_balance_loss', type=click.Choice(['bce', 'log', 'bug']), default='bug',
              help='Class balancing loss function')
@click.option('--learning_rate', type=float, default=1e-4, help='learning rate (Adam)')
@click.option('--pretrained_lr_factor', type=float, default=0.1,
              help='learning rate scale factor for pre-trained layers')
@click.option('--fix_layers', type=str, default='',
              help='List of layers to fix')
@click.option('--double_softmax', is_flag=True, default=False)
@click.option('--use_dropout', is_flag=True, default=False)
@click.option('--src_scale_u_range', type=str, default='',
              help='aug xform: scale uniform range; lower:upper')
@click.option('--src_scale_x_range', type=str, default='',
              help='aug xform: scale x range; lower:upper')
@click.option('--src_scale_y_range', type=str, default='',
              help='aug xform: scale y range; lower:upper')
@click.option('--src_affine_std', type=float, default=0.1, help='aug xform: random affine transform std-dev')
@click.option('--src_xlat_range', type=float, default=0.0, help='aug xform: translation range')
@click.option('--src_rot_std', type=float, default=0.2, help='aug xform: rotation std-dev')
@click.option('--src_hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--src_intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--src_colour_rot_std', type=float, default=0.05,
              help='aug colour; colour rotation standard deviation')
@click.option('--src_colour_off_std', type=float, default=0.1,
              help='aug colour; colour offset standard deviation')
@click.option('--src_greyscale', is_flag=True, default=False,
              help='aug colour; enable greyscale')
@click.option('--src_cutout_prob', type=float, default=0.0,
              help='aug cutout probability')
@click.option('--src_cutout_size', type=float, default=0.3,
              help='aug cutout size (fraction)')
@click.option('--tgt_scale_u_range', type=str, default='',
              help='aug xform: scale uniform range; lower:upper')
@click.option('--tgt_scale_x_range', type=str, default='',
              help='aug xform: scale x range; lower:upper')
@click.option('--tgt_scale_y_range', type=str, default='',
              help='aug xform: scale y range; lower:upper')
@click.option('--tgt_affine_std', type=float, default=0.1, help='aug xform: random affine transform std-dev')
@click.option('--tgt_xlat_range', type=float, default=0.0, help='aug xform: translation range')
@click.option('--tgt_rot_std', type=float, default=0.2, help='aug xform: rotation std-dev')
@click.option('--tgt_hflip', default=False, is_flag=True, help='aug xform: enable random horizontal flips')
@click.option('--tgt_intens_scale_range', type=str, default='',
              help='aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)')
@click.option('--tgt_colour_rot_std', type=float, default=0.05,
              help='aug colour; colour rotation standard deviation')
@click.option('--tgt_colour_off_std', type=float, default=0.1,
              help='aug colour; colour offset standard deviation')
@click.option('--tgt_greyscale', is_flag=True, default=False,
              help='aug colour; enable greyscale')
@click.option('--tgt_cutout_prob', type=float, default=0.0,
              help='aug cutout probability')
@click.option('--tgt_cutout_size', type=float, default=0.3,
              help='aug cutout size (fraction)')
@click.option('--constrain_crop', type=int, default=0,
              help='random cropping; constrain crop pairs to be within this distance (-1 = off)')
@click.option('--img_pad_width', type=int, default=0,
              help='random cropping; padding width')
@click.option('--num_epochs', type=int, default=200, help='number of epochs')
@click.option('--batch_size', type=int, default=64, help='mini-batch size')
@click.option('--epoch_size', type=str, default='target',
              help='# of samples in epoch; either a number or \'source\' or \'target\'')
@click.option('--seed', type=int, default=0, help='random seed (0 for time-based)')
@click.option('--log_file', type=str, default='', help='log file path (none to disable)')
@click.option('--skip_epoch_eval', is_flag=True, default=False)
@click.option('--result_file', type=str, default='',
              help='path to HFD5 file to save results to')
@click.option('--record_history', is_flag=True, default=False, help='Record per-epoch target prediction history')
@click.option('--model_file', type=str, default='',
              help='path to file to save model to')
@click.option('--hide_progress_bar', is_flag=True, default=False, help='Hide training progress bar')
@click.option('--subsetsize', type=int, default=0, help='Subset size (0 to use full dataset)')
@click.option('--subsetseed', type=int, default=0, help='subset random seed (0 for time based)')
@click.option('--device', type=int, default=0, help='Device')
@click.option('--num_threads', type=int, default=2, help='Number of worker threads')
def experiment(exp, arch, rnd_init, img_size, confidence_thresh, teacher_alpha, unsup_weight,
               cls_balance, cls_balance_loss,
               learning_rate, pretrained_lr_factor, fix_layers,
               double_softmax, use_dropout,
               src_scale_u_range, src_scale_x_range, src_scale_y_range,
               src_affine_std, src_xlat_range, src_rot_std, src_hflip,
               src_intens_scale_range, src_colour_rot_std, src_colour_off_std, src_greyscale,
               src_cutout_prob, src_cutout_size,
               tgt_scale_u_range, tgt_scale_x_range, tgt_scale_y_range,
               tgt_affine_std, tgt_xlat_range, tgt_rot_std, tgt_hflip,
               tgt_intens_scale_range, tgt_colour_rot_std, tgt_colour_off_std, tgt_greyscale,
               tgt_cutout_prob, tgt_cutout_size,
               constrain_crop, img_pad_width,
               num_epochs, batch_size, epoch_size, seed,
               log_file, skip_epoch_eval, result_file, record_history, model_file, hide_progress_bar,
               subsetsize, subsetseed,
               device, num_threads):
    settings = locals().copy()

    if rnd_init:
        if fix_layers != '':
            print('`rnd_init` and `fix_layers` are mutually exclusive')
            return

    if epoch_size not in {'source', 'target'}:
        try:
            epoch_size = int(epoch_size)
        except ValueError:
            print('epoch_size should be an integer, \'source\', or \'target\', not {}'.format(epoch_size))
            return

    import os
    import sys
    import pickle
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

    src_intens_scale_range_lower, src_intens_scale_range_upper = cmdline_helpers.colon_separated_range(src_intens_scale_range)
    tgt_intens_scale_range_lower, tgt_intens_scale_range_upper = cmdline_helpers.colon_separated_range(tgt_intens_scale_range)
    src_scale_u_range = cmdline_helpers.colon_separated_range(src_scale_u_range)
    tgt_scale_u_range = cmdline_helpers.colon_separated_range(tgt_scale_u_range)
    src_scale_x_range = cmdline_helpers.colon_separated_range(src_scale_x_range)
    tgt_scale_x_range = cmdline_helpers.colon_separated_range(tgt_scale_x_range)
    src_scale_y_range = cmdline_helpers.colon_separated_range(src_scale_y_range)
    tgt_scale_y_range = cmdline_helpers.colon_separated_range(tgt_scale_y_range)


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
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F
    import optim_weight_ema

    if hide_progress_bar:
        progress_bar = None
    else:
        progress_bar = tqdm.tqdm


    with torch.cuda.device(device):
        pool = work_pool.WorkerThreadPool(num_threads)

        n_chn = 0
        half_batch_size = batch_size // 2

        if arch == '':
            if exp in {'train_val', 'train_test'}:
                arch = 'resnet50'

        if rnd_init:
            mean_value = np.array([0.5, 0.5, 0.5])
            std_value = np.array([0.5, 0.5, 0.5])
        else:
            mean_value = np.array([0.485, 0.456, 0.406])
            std_value = np.array([0.229, 0.224, 0.225])


        img_shape = (img_size, img_size)
        img_padding = (img_pad_width, img_pad_width)

        if exp == 'visda_train_val':
            d_source = visda17_dataset.TrainDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = visda17_dataset.ValidationDataset(img_size=img_shape, range01=True, rgb_order=True)
        elif exp == 'visda_train_test':
            d_source = visda17_dataset.TrainDataset(img_size=img_shape, range01=True, rgb_order=True)
            d_target = visda17_dataset.TestDataset(img_size=img_shape, range01=True, rgb_order=True)

            if not skip_epoch_eval:
                print('WARNING: setting skip_epoch_eval to True')
                skip_epoch_eval = True
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

        # Tensorboard log

        # Subset
        source_indices, target_indices, n_src, n_tgt = image_dataset.subset_indices(
            d_source, d_target, subsetsize, subsetseed
        )


        #
        # Result file
        #

        if result_file != '':
            h5_filters = tables.Filters(complevel=9, complib='blosc')
            f_target_pred = tables.open_file(result_file, mode='w')
            g_tgt_pred = f_target_pred.create_group(f_target_pred.root, 'target_pred_y', 'Target prediction')
            if record_history:
                arr_tgt_pred_history = f_target_pred.create_earray(g_tgt_pred, 'y_prob_history', tables.Float32Atom(),
                                                           (0, n_tgt, d_target.n_classes),
                                                           filters=h5_filters)
            else:
                arr_tgt_pred_history = None
        else:
            arr_tgt_pred_history = None
            f_target_pred = None
            g_tgt_pred = None


        n_classes = d_source.n_classes

        print('Loaded data')

        net_class = network_architectures.get_build_fn_for_architecture(arch)

        student_net = net_class(n_classes, img_size, use_dropout, not rnd_init).cuda()
        teacher_net = net_class(n_classes, img_size, use_dropout, not rnd_init).cuda()
        student_params = list(student_net.parameters())
        teacher_params = list(teacher_net.parameters())
        for param in teacher_params:
            param.requires_grad = False

        if rnd_init:
            new_student_optimizer = torch.optim.Adam(student_params, lr=learning_rate)
            pretrained_student_optimizer = None
        else:
            named_params = list(student_net.named_parameters())
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

            new_student_optimizer = torch.optim.Adam(new_params, lr=learning_rate)
            if len(pretrained_params) > 0:
                pretrained_student_optimizer = torch.optim.Adam(pretrained_params, lr=learning_rate * pretrained_lr_factor)
            else:
                pretrained_student_optimizer = None
        teacher_optimizer = optim_weight_ema.WeightEMA(teacher_params, student_params, alpha=teacher_alpha)
        classification_criterion = nn.CrossEntropyLoss()

        print('Built network')


        # Image augmentation

        src_aug = augmentation.ImageAugmentation(
            src_hflip, src_xlat_range, src_affine_std, rot_std=src_rot_std,
            intens_scale_range_lower=src_intens_scale_range_lower, intens_scale_range_upper=src_intens_scale_range_upper,
            colour_rot_std=src_colour_rot_std, colour_off_std=src_colour_off_std, greyscale=src_greyscale,
            scale_u_range=src_scale_u_range, scale_x_range=src_scale_x_range, scale_y_range=src_scale_y_range,
            cutout_probability = src_cutout_prob, cutout_size = src_cutout_size)

        tgt_aug = augmentation.ImageAugmentation(
            tgt_hflip, tgt_xlat_range, tgt_affine_std, rot_std=tgt_rot_std,
            intens_scale_range_lower=tgt_intens_scale_range_lower, intens_scale_range_upper=tgt_intens_scale_range_upper,
            colour_rot_std=tgt_colour_rot_std, colour_off_std=tgt_colour_off_std, greyscale=tgt_greyscale,
            scale_u_range=tgt_scale_u_range, scale_x_range=tgt_scale_x_range, scale_y_range=tgt_scale_y_range,
            cutout_probability=tgt_cutout_prob, cutout_size=tgt_cutout_size)

        test_aug = augmentation.ImageAugmentation(
            tgt_hflip, tgt_xlat_range, 0.0, rot_std=0.0,
            scale_u_range=tgt_scale_u_range, scale_x_range=tgt_scale_x_range, scale_y_range=tgt_scale_y_range)

        border_value = int(np.mean(mean_value) * 255 + 0.5)

        sup_xf = image_transforms.Compose(
            image_transforms.ScaleCropAndAugmentAffine(img_shape, img_padding, True, src_aug, border_value, mean_value,
                                                       std_value),
            image_transforms.ToTensor(),
        )

        if constrain_crop >= 0:
            unsup_xf = image_transforms.Compose(
                image_transforms.ScaleCropAndAugmentAffinePair(
                    img_shape, img_padding, constrain_crop, True, tgt_aug, border_value, mean_value, std_value),
                image_transforms.ToTensor(),
            )
        else:
            unsup_xf = image_transforms.Compose(
                image_transforms.ScaleCropAndAugmentAffine(img_shape, img_padding, True, tgt_aug, border_value,
                                                           mean_value, std_value),
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


        if constrain_crop >= 0:
            def augment(X_sup, y_sup, X_tgt):
                X_sup = sup_xf(X_sup)[0]
                X_unsup_both = unsup_xf(X_tgt)[0]
                X_unsup_stu = X_unsup_both[:len(X_tgt)]
                X_unsup_tea = X_unsup_both[len(X_tgt):]
                return X_sup, y_sup, X_unsup_stu, X_unsup_tea
        else:
            def augment(X_sup, y_sup, X_tgt):
                X_sup = sup_xf(X_sup)[0]
                X_unsup_stu = unsup_xf(X_tgt)[0]
                X_unsup_tea = unsup_xf(X_tgt)[0]
                return X_sup, y_sup, X_unsup_stu, X_unsup_tea


        cls_bal_fn = network_architectures.get_cls_bal_function(cls_balance_loss)

        def compute_aug_loss(stu_out, tea_out):
            # Augmentation loss
            conf_tea = torch.max(tea_out, 1)[0]
            conf_mask = torch.gt(conf_tea, confidence_thresh).float()

            d_aug_loss = stu_out - tea_out
            aug_loss = d_aug_loss * d_aug_loss

            aug_loss = torch.mean(aug_loss, 1) * conf_mask

            # Class balance loss
            if cls_balance > 0.0:
                # Average over samples to get average class prediction
                avg_cls_prob = torch.mean(stu_out, 0)
                # Compute loss
                equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))

                equalise_cls_loss = torch.mean(equalise_cls_loss) * n_classes

                equalise_cls_loss = equalise_cls_loss * torch.mean(conf_mask, 0)
            else:
                equalise_cls_loss = None

            return aug_loss, conf_mask, equalise_cls_loss

        _one = torch.autograd.Variable(torch.from_numpy(np.array([1.0]).astype(np.float32)).cuda())
        def f_train(X_sup, y_sup, X_unsup0, X_unsup1):
            X_sup = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            y_sup = torch.autograd.Variable(torch.from_numpy(y_sup).long().cuda())
            X_unsup0 = torch.autograd.Variable(torch.from_numpy(X_unsup0).cuda())
            X_unsup1 = torch.autograd.Variable(torch.from_numpy(X_unsup1).cuda())

            if pretrained_student_optimizer is not None:
                pretrained_student_optimizer.zero_grad()
            new_student_optimizer.zero_grad()
            student_net.train(mode=True)
            teacher_net.train(mode=True)

            sup_logits_out = student_net(X_sup)
            student_unsup_logits_out = student_net(X_unsup0)
            student_unsup_prob_out = F.softmax(student_unsup_logits_out)
            teacher_unsup_logits_out = teacher_net(X_unsup1)
            teacher_unsup_prob_out = F.softmax(teacher_unsup_logits_out)

            # Supervised classification loss
            if double_softmax:
                clf_loss = classification_criterion(F.softmax(sup_logits_out), y_sup)
            else:
                clf_loss = classification_criterion(sup_logits_out, y_sup)

            aug_loss, conf_mask, cls_bal_loss = compute_aug_loss(student_unsup_prob_out, teacher_unsup_prob_out)

            conf_mask_count = torch.sum(conf_mask)

            unsup_loss = torch.mean(aug_loss)
            loss_expr = clf_loss + unsup_loss * unsup_weight
            if cls_bal_loss is not None:
                loss_expr = loss_expr + cls_bal_loss * cls_balance * unsup_weight

            loss_expr.backward()
            if pretrained_student_optimizer is not None:
                pretrained_student_optimizer.step()
            new_student_optimizer.step()
            teacher_optimizer.step()

            n_samples = X_sup.size()[0]

            mask_count = conf_mask_count.data.cpu()[0]

            outputs = [float(clf_loss.data.cpu()[0]) * n_samples,
                       float(unsup_loss.data.cpu()[0]) * n_samples,
                       mask_count]
            return tuple(outputs)

        print('Compiled training function')

        def f_pred_src(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            teacher_net.train(mode=False)
            return (F.softmax(teacher_net(X_var)).data.cpu().numpy(),)

        def f_pred_tgt(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            teacher_net.train(mode=False)
            return (F.softmax(teacher_net(X_var)).data.cpu().numpy(),)

        def f_pred_tgt_mult(X_sup):
            teacher_net.train(mode=False)
            y_pred_aug = []
            for aug_i in range(len(X_sup)):
                X_var = torch.autograd.Variable(torch.from_numpy(X_sup[aug_i, ...]).cuda())
                y_pred = F.softmax(teacher_net(X_var)).data.cpu().numpy()
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

        # Report setttings
        log('Program = {}'.format(sys.argv[0]))
        log('Settings: {}'.format(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))])))

        # Report dataset size
        log('Dataset:')
        log('SOURCE len(X)={}, y.shape={}'.format(len(d_source.images), d_source.y.shape))
        log('TARGET len(X)={}'.format(len(d_target.images)))


        if epoch_size == 'source':
            n_samples = n_src
        elif epoch_size == 'target':
            n_samples = n_tgt
        else:
            n_samples = epoch_size
        n_train_batches = n_samples // batch_size
        n_test_batches = n_tgt // (batch_size * 2) + 1

        print('Training...')
        sup_ds = data_source.ArrayDataSource([d_source.images, d_source.y], repeats=-1, indices=source_indices)
        tgt_train_ds = data_source.ArrayDataSource([d_target.images], repeats=-1, indices=target_indices)
        train_ds = data_source.CompositeDataSource([sup_ds, tgt_train_ds]).map(augment)
        train_ds = pool.parallel_data_source(train_ds, batch_buffer_size=min(20, n_train_batches))

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


        best_mask_rate = 0.0
        best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}

        train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

        for epoch in range(num_epochs):
            t1 = time.time()

            if not skip_epoch_eval:
                test_batch_iter = target_test_ds.batch_iterator(batch_size=batch_size * 2)
            else:
                test_batch_iter = None

            train_clf_loss, train_unsup_loss, mask_rate = data_source.batch_map_mean(
                f_train, train_batch_iter, progress_iter_func=progress_bar, n_batches=n_train_batches)

            # train_clf_loss, train_unsup_loss, mask_rate = train_ds.batch_map_mean(
            #     f_train, batch_size=batch_size, shuffle=shuffle_rng, n_batches=n_train_batches,
            #     progress_iter_func=progress_bar)


            if mask_rate > best_mask_rate:
                best_mask_rate = mask_rate
                improve = True
                improve_str = '*** '
                best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
            else:
                improve = False
                improve_str = ''


            if not skip_epoch_eval:
                tgt_pred_prob_y, = data_source.batch_map_concat(f_pred_tgt, test_batch_iter,
                                                                progress_iter_func=progress_bar)
                mean_class_acc, cls_acc_str = evaluator.evaluate(tgt_pred_prob_y)
                t2 = time.time()

                log('{}Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}, unsup loss={:.6f}, mask={:.3%}; '
                    'TGT mean class acc={:.3%}'.format(
                    improve_str, epoch, t2 - t1, train_clf_loss, train_unsup_loss, mask_rate, mean_class_acc))
                log('  per class:  {}'.format(cls_acc_str))

                # Save results
                if arr_tgt_pred_history is not None:
                    arr_tgt_pred_history.append(tgt_pred_prob_y[None, ...].astype(np.float32))
            else:
                t2 = time.time()
                log('{}Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}, unsup loss={:.6f}, mask={:.3%}'.format(
                    improve_str, epoch, t2 - t1, train_clf_loss, train_unsup_loss, mask_rate))


        # Save network
        if model_file != '':
            with open(model_file, 'wb') as f:
                pickle.dump(best_teacher_model_state, f)

        # Restore network to best state
        teacher_net.load_state_dict({k: torch.from_numpy(v) for k, v in best_teacher_model_state.items()})

        # Predict on test set, without augmentation
        tgt_pred_prob_y, = target_test_ds.batch_map_concat(f_pred_tgt, batch_size=batch_size,
                                                           progress_iter_func=progress_bar)

        if d_target.has_ground_truth:
            mean_class_acc, cls_acc_str = evaluator.evaluate(tgt_pred_prob_y)

            log('FINAL: TGT mean class acc={:.3%}'.format(mean_class_acc))
            log('  per class:  {}'.format(cls_acc_str))

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