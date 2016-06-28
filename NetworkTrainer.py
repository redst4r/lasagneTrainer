import time
import matplotlib.pyplot as plt
from batchgenerators import iterate_minibatches
import lasagne.layers
import theano
import numpy as np
import progressbar
from nolearn.lasagne.util import ansi
from collections import namedtuple

timeinfo = namedtuple('timeinfo', 'total train_gpu train_batch val_gpu val_batch')
# ------------------------------------------------------------------

class NetworkTrainer(object):
    """trains a given lasagne network using gradient desend"""
    def __init__(self, network, quiet=False):
        """
        network: last layer of the lasagne network
        """
        self.network = network

        # save the training/validation etc across training runs
        self.trainError, self.trainAccuracy, self.valError, self.valAccuracy = [], [], [], []
        self.W_array = []
        self.quiet = quiet
        self.best_network_params = None

    def print_layers(self):
        layers_list = lasagne.layers.get_all_layers(self.network)

        for l in layers_list:
            print(l.name)

    def get_network(self):
        return self.network

    def do_training_with_generators(self, training_generator, validation_generator, trainInput, valInput, train_fn, val_fn, epochs):
        """
        train the network. uses generator_factories (a call to the factory will give a generator that goes through the data once, maybe in batches)
        :param training_generator:  function f(X,y) that creates a generator which yields tuples of (X,y) batches
        :param validation_generator:  --"--
        :param trainInput: feeds into the training generator. in fact it is called as training_generator(*trainInput)
        :param valInput:  --"--
        :param train_fn: theano.function to train the network
        :param val_fn: theano.function to validate the network
        :param epochs: number of epochs to run
        :return:
        """
        # calling training_generator(trainInput) has to create a generator that loops over the data once!
        # calling it again must provide another generator that again iterates over the data
        if not self.quiet:
            print("Epoch\t\tTrLoss\t\tTrAcc\t\tValLoss\t\tvalAcc\t\tTime (trGPU trBa vaGPU vaBa)")

        for epoch in range(epochs):
            train_err, train_batches, train_acc = 0, 0, 0

            # keep track of the timings, what takes how long
            start_time_total = time.time()
            start_time_train = time.time()

            trainFun_time = 0  # how long do the evals of the training functions take
            for batch in training_generator(*trainInput):
                inputs, targets = batch

                tmp_time_train = time.time()
                terr, tacc = train_fn(inputs, targets)
                train_err += terr
                train_acc += tacc
                trainFun_time += time.time() - tmp_time_train
                train_batches += 1

            dt_train = time.time() - start_time_train  # this is trainFun_time + time to load the batches
            dt_train_gpu = trainFun_time  # the amount spent evaluating the network
            dt_train_batchload = dt_train-trainFun_time  # time spent loadin the batches

            # validation
            val_err, val_acc, val_batches = 0, 0, 0

            start_time_val= time.time()
            val_fnFun_time = 0  # how long do the evals of the valdiation functions take
            for batch in validation_generator(*valInput):
                inputs, targets = batch
                tmp_time_val = time.time() # TODO replace with timing context manger?!
                err, acc = val_fn(inputs, targets)
                val_fnFun_time += time.time() - tmp_time_val

                val_err += err
                val_acc += acc
                val_batches += 1

            dt_val = time.time() - start_time_val
            dt_val_gpu = val_fnFun_time
            dt_val_batchload = dt_val - dt_val_gpu

            dt_total = time.time() - start_time_total

            the_timing = timeinfo(total=dt_total, train_gpu=dt_train_gpu,
                         train_batch=dt_train_batchload,
                         val_gpu=dt_val_gpu, val_batch=dt_val_batchload )

            self._after_epoch_helper(train_err, train_acc, val_err, val_acc, train_batches, val_batches, the_timing, epoch)

    def doTraining(self, trainData, trainLabels, valData, valLabels, train_fn, val_fn, pred_fn, epochs, batchsize):
        """
        train_fn, val_fn are theano.functions
        """
        # TODO UNTESTED (not even run before)
        def gen_factory(x,y):
            return iterate_minibatches(x, y, batchsize, shuffle=True)

        self.do_training_with_generators(gen_factory, gen_factory,
                                         (trainData, trainLabels),
                                         (valData, valLabels),
                                         train_fn, val_fn, epochs)

    def _after_epoch_helper(self, train_err, train_acc, val_err, val_acc, train_batches, val_batches, the_timing, epoch):
        """
        convenience fuction used by both doTraining and doTraining_with_generators
        :param train_err:
        :param val_err:
        :param val_acc:
        :param train_batches:
        :param val_batches:
        :param the_timing: namedtuple of type 'timeinfo'
        :param epoch:
        :return:
        """
        # normalize the error to batches
        tr_loss_normed = train_err / train_batches
        tr_acc_normed = train_acc / train_batches * 100
        val_loss_normed = val_err / val_batches
        val_acc_normed = val_acc / val_batches * 100

        if not self.quiet:
            self._print_epoch_summary(tr_loss_normed, tr_acc_normed, val_loss_normed, val_acc_normed, the_timing, epoch)

        # record the parametesr if new best network
        val_loss_best = all([val_loss_normed < _ for _ in self.valError])
        if val_loss_best or not self.best_network_params:
            # new best performer or no other present yet
            self.best_network_params = lasagne.layers.get_all_param_values(self.network)

        # record for later
        self.trainError.append(tr_loss_normed)
        self.trainAccuracy.append(tr_acc_normed)
        self.valError.append(val_loss_normed)
        self.valAccuracy.append(val_acc_normed)

    def _print_epoch_summary(self, tr_loss_normed, tr_acc_normed, val_loss_normed, val_acc_normed, the_timing, epoch ):
        """
        does a pretty print of the epoch results in terms of train/val error and time taken.
        also colors the best errors
        :param tr_loss_normed:
        :param val_loss_normed:
        :param val_acc_normed:
        :param epoch:
        :return:
        """
        tr_loss_best = all([tr_loss_normed < _ for _ in self.trainError])  # flag for best seen training error yet
        tr_loss_str = "{}{:.3f}{}".format(ansi.CYAN if tr_loss_best else "",
                                          tr_loss_normed,
                                          ansi.ENDC if tr_loss_best else "")
        tr_acc_str = "{}{:.3f}{}".format(ansi.CYAN if tr_loss_best else "",
                                         tr_acc_normed,
                                         ansi.ENDC if tr_loss_best else "")

        val_loss_best = all([val_loss_normed < _ for _ in self.valError])
        val_loss_str = "{}{:.3f}{}".format(ansi.GREEN if val_loss_best else "",
                                           val_loss_normed,
                                           ansi.ENDC if val_loss_best else "")

        val_acc_best = all([val_acc_normed > _ for _ in self.valAccuracy])
        val_acc_str = "{}{:.3f}{}".format(ansi.MAGENTA if val_acc_best else "",
                                          val_acc_normed,
                                          ansi.ENDC if val_acc_best else "")

        time_string = "%.3f (%.1f + %.1f + %.1f + %.1f)" % (the_timing.total, the_timing.train_gpu, the_timing.train_batch, the_timing.val_gpu, the_timing.val_batch)
        prettyString = "%d\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s" % (epoch, tr_loss_str, tr_acc_str, val_loss_str, val_acc_str, time_string)
        print(prettyString)

    def predict(self, X, batchsize=None):
        """
        use the trained network to predict scores for given samples

        its a bit tricky since we first hve to create a prediction theano-function.
        this is cached in the attribute ._pred_fn, since it tae=kes a while to compute
        :param X:
        :return:
        """

        if not hasattr(self, 'pred_fn'):
            prediction = lasagne.layers.get_output(self.network, deterministic=True)
            input_layer = lasagne.layers.get_all_layers(self.network)[0]  # kind of hacky, get the input variable of the net
            input_var = input_layer.input_var
            assert isinstance(input_layer, lasagne.layers.InputLayer)
            self.pred_fn = theano.function([input_var], prediction)

        # prediction, but in batches (mem overflow!)
        # TODO replace with batch iterator
        if batchsize is None:
            return self.pred_fn(X)
        else:
            scores = []
            bar = progressbar.ProgressBar()
            for start in bar(np.arange(0, X.shape[0], batchsize)):
                thebatch = X[start:(start + batchsize)]
                scores.append(self.pred_fn(thebatch))
            return np.vstack(scores)

    def plot_error(self):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.trainError, 'b-')
        plt.plot(self.valError, 'g-')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "val"])

        plt.subplot(2, 1, 2)
        plt.plot(self.trainAccuracy, 'b-')
        plt.plot(self.valAccuracy, 'g-')
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
