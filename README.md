# Self-ensembling for visual domain adaptation (photos)

Implementation of the paper [Self-ensembling for visual domain adaptation](https://arxiv.org/abs/1706.05208),
accepted as a poster at ICLR 2018.

For photographic datasets such as The VisDA visual domain adaptation challenge and the Office dataset.

For the small image experiments go to
[https://github.com/Britefury/self-ensemble-visual-domain-adapt/](https://github.com/Britefury/self-ensemble-visual-domain-adapt/).


## Setup

You need to specify where your datasets can be found on your machine;cCreate a file called `datasets.cfg` that
contains the following (note that you can skip e.g. the Office paths if you don't intend to run the Office
experiments):

```
[paths]
visda17_clf_train=<path_to_visda_train>
visda17_clf_validation=<path_to_visda_validation>
visda17_clf_test=<path_to_visda_test>
office_amazon=<path_to_office_amazon>
office_dslr=<path_to_office_dslr>
office_webcam=<path_to_office_webcam>
```


## Running the experiments

The main experiment file is `experiment_selfens_meanteacher.py`. Invoking it with the `--help` command line argument
will show you the command line options.

To replicate our results, invoke the shell scripts.

To train a ResNet-152 based network on the training and validation sets:
```sh run_visda17_trainval_resnet152.sh <GPU> <RUN>```

Where `<GPU>` is an integer identifying the GPU to use and <RUN> enumerates the experiment number so that
you can keep logs of multiple repeated runs separate, e.g.:

```sh run_visda17_trainval_resnet152.sh 0 01```
Will run on GPU 0 and will generate log files with names suffixed with `run01`.


To train a ResNet-152 based network on the training and validation sets using basic augmentation:
```sh run_visda17_trainval_resnet152_basicaug.sh <GPU> <RUN>```

To train a ResNet-152 based network on the training and validation sets using minimal augmentation:
```sh run_visda17_trainval_resnet152_minaug.sh <GPU> <RUN>```

To train a ResNet-50 based network on the training and validation sets:
```sh run_visda17_trainval_resnet50.sh <GPU> <RUN>```

To train a ResNet-152 based network on the training and validation sets using minimal augmentation using supervised
training only:
```sh run_visda17_trainval_sup_resnet152_minaug.sh <GPU> <RUN>```

There are similar experiments for the training and test set.

Running the experiments will generate 3 output files; a log file, a prediction file and a model file. The log file
will contain a copy of the training output that was printed to the console. The prediction will will contain the
history of predictions generated during training. The model file will contain the network weights.

To generate a submission file suitable for submission to the VisDA-17 CodaLab site, use the program
`build_visda_submission.py`.

## Note on bugs

There are two 'bugs' that are used in our VisDA experiment.

1. Due to a programming error, we applied the softmax non-linearity twice to the end of the network. When we fixed
this our performance on the validation set dropped, so it was re-introduced as a command line option that is used
for our experiments.
2. Due to a programming error, the class balance loss binary cross entropy was implemented as
-(p*log(q) + (1-p)*log(1+q)) instead of -(p*log(q) + (1-p)*log(1-q)). Once again, fixing this dropped our performance
so the command line option --cls_balance_loss takes the value 'bug' to use it. This is used in our experiments.

## Installation

You will need:

- Python 3.6 (Anaconda Python recommended)
- OpenCV with Python bindings
- PyTorch

First, install OpenCV and PyTorch as `pip` may have trouble with these.

### OpenCV with Python bindings

On Linux, install using `conda`:

```> conda install opencv```

On Windows, go to [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and
download the OpenCV wheel file and install with:

```> pip install <path_of_opencv_file>```

### PyTorch

On Linux:

```> conda install pytorch torchvision -c pytorch```

On Windows:

```> conda install -c peterjc123 pytorch cuda90```

### The rest

Use pip like so:

```> pip install -r requirements.txt```
