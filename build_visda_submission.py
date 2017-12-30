import click


@click.command()
@click.argument('output', type=click.File('wb'))
@click.argument('source_pred_path', type=click.Path(exists=True))
@click.argument('predictions_paths', type=click.Path(exists=True), nargs=-1)
@click.option('--no_aug', is_flag=True, default=False)
@click.option('--epoch', type=int, default=None)
def build_submission(output, source_pred_path, predictions_paths, no_aug, epoch):
    import os
    import numpy as np
    import tables
    import zipfile


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


    def write_predictions(dest_file, src_paths):
        if len(src_paths) == 1 and os.path.splitext(src_paths[0])[1] == '.txt':
            data = open(src_paths[0], 'rb').read()
            dest_file.write(data)
        else:
            best_ys = [get_best_y(path) for path in src_paths]

            # Average
            best_y = np.mean(np.array(best_ys), axis=0)

            pred_y_cls = np.argmax(best_y, axis=1)

            print('Obtained predictions')


            for i in range(len(pred_y_cls)):
                dest_file.write('{}\n'.format(pred_y_cls[i]).encode('us-ascii'))

    z = zipfile.ZipFile(output, mode='w')
    write_predictions(z.open('source_results.txt', 'w'), [source_pred_path])
    write_predictions(z.open('adaptation_results.txt', 'w'), predictions_paths)
    z.close()


if __name__ == '__main__':
    build_submission()