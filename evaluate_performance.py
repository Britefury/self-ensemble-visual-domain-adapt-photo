import click


@click.command()
@click.argument('exp', type=click.Choice([
    'visda_train_val', 'visda_train_test',
    'office_amazon_dslr', 'office_amazon_webcam', 'office_dslr_amazon', 'office_dslr_webcam', 'office_webcam_amazon', 'office_webcam_dslr',
    ]))
@click.argument('predictions_path', type=click.Path(exists=True))
@click.option('--metric', type=click.Choice(['acc', 'class_acc']), default='class_acc')
@click.option('--aug', is_flag=True, default=False)
def build_submission(exp, predictions_path, metric, aug):
    import numpy as np
    import tables
    import visda17_dataset, office_dataset

    img_shape=(160,160)

    if exp == 'visda_train_val':
        d_target = visda17_dataset.ValidationDataset(img_size=img_shape, range01=True, rgb_order=True)
    elif exp == 'visda_train_test':
        d_target = visda17_dataset.TestDataset(img_size=img_shape, range01=True, rgb_order=True)
    elif exp in {'office_amazon_dslr', 'office_webcam_dslr'}:
        d_target = office_dataset.OfficeDSLRDataset(img_size=img_shape, range01=True, rgb_order=True)
    elif exp in {'office_amazon_webcam', 'office_dslr_webcam'}:
        d_target = office_dataset.OfficeWebcamDataset(img_size=img_shape, range01=True, rgb_order=True)
    elif exp in {'office_dslr_amazon', 'office_webcam_amazon'}:
        d_target = office_dataset.OfficeAmazonDataset(img_size=img_shape, range01=True, rgb_order=True)
    else:
        print('Unknown experiment type \'{}\''.format(exp))
        return


    if not d_target.has_ground_truth:
        print('Cannot evaluate: target dataset has no ground truth available.')
        return



    def get_y(path):
        print('Reading {}'.format(path))
        f_target_pred = tables.open_file(path, mode='r')

        if aug:
            best_y = f_target_pred.root.target_pred_y.y_prob_aug
        else:
            best_y = f_target_pred.root.target_pred_y.y_prob

        return best_y


    y = get_y(predictions_path)


    print('Loaded predictions')

    if metric == 'class_acc':
        evaluator = d_target.prediction_evaluator()
        mean_class_acc, cls_acc_str = evaluator.evaluate(y)
        print('Mean class acc={:.3%}'.format(mean_class_acc))
        print(cls_acc_str)
    elif metric == 'acc':
        acc = (np.argmax(y, axis=1) == d_target.y).astype(float).mean()
        print('Acc={:.3%}'.format(acc))



if __name__ == '__main__':
    build_submission()