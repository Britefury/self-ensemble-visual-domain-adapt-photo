import click


@click.command()
@click.argument('output', type=click.File('wb'))
@click.argument('predictions_paths', type=click.Path(exists=True), nargs=-1)
@click.option('--no_aug', is_flag=True, default=False)
@click.option('--epoch', type=int, default=None)
def build_submission(output, predictions_paths, no_aug, epoch):
    import numpy as np
    import tables


    def get_best_y(path):
        print('Reading {}'.format(path))
        f_target_pred = tables.open_file(path, mode='r')

        if epoch is not None:
            y_prob_history = f_target_pred.root.target_pred_y.y
            return y_prob_history[epoch]

        if no_aug:
            return f_target_pred.root.target_pred_y.y_prob
        else:
            return f_target_pred.root.target_pred_y.y_prob_aug


    best_ys = [get_best_y(path) for path in predictions_paths]


    print('Loaded predictions')

    # Average
    best_y = np.mean(np.array(best_ys), axis=0)

    pred_y_cls = np.argmax(best_y, axis=1)

    print('Obtained predictions')


    for i in range(len(pred_y_cls)):
        output.write('{}\n'.format(pred_y_cls[i]).encode('us-ascii'))




if __name__ == '__main__':
    build_submission()