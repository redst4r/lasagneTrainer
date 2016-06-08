import time
import matplotlib.pyplot as plt
from batchgenerators import iterate_minibatches
import lasagne.layers
import theano
import numpy as np
import progressbar
from nolearn.lasagne.util import ansi
# ------------------------------------------------------------------

class NetworkTrainer(object):
    """trains a given lasagne network using gradient desend"""
    def __init__(self, network):
        """
        network: last layer of the lasagne network
        """
        self.network = network

        # save the training/validation etc across training runs
        self.trainError, self.valError, self.valAccuracy = [], [], []
        self.W_array = []

        self.best_network_params = None

    def print_layers(self):
        layers_list = lasagne.layers.get_all_layers(self.network)

        for l in layers_list:
            print(l.name)

    def get_network(self):
        return self.network

    def do_training_with_generators(self, training_generator, validation_generator, trainInput, valInput, train_fn, val_fn, epochs):

        # calling training_generator(trainInput) has to create a generator that loops over the data once!
        # calling it again must provide another generator that again iterates over the data
        print("Epoch\t\tTrLoss\t\tValLoss\t\tvalAcc\t\tTime")
        for epoch in range(epochs):
            train_err, train_batches = 0, 0
            start_time = time.time()
            for batch in training_generator(*trainInput):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # validation
            val_err, val_acc, val_batches = 0, 0, 0
            for batch in validation_generator(*valInput):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            dt = time.time() - start_time
            self._after_epoch_helper(train_err, val_err, val_acc, train_batches, val_batches, dt, epoch)

    def doTraining(self, trainData, trainLabels, valData, valLabels, train_fn, val_fn, pred_fn, epochs, batchsize):
        """
        train_fn, val_fn are theano.functions
        """
        # TODO this looks like a special case of doTraining_with_generators -> GENERALIZE
        print("Epoch\t\tTrLoss\t\tValLoss\t\tvalAcc\t\tTime")
        for epoch in range(epochs):
            # training
            train_err, train_batches = 0, 0
            start_time = time.time()
            for batch in iterate_minibatches(trainData, trainLabels, batchsize, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # validation
            val_err, val_acc, val_batches = 0, 0, 0
            for batch in iterate_minibatches(valData, valLabels, batchsize, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            dt = time.time() - start_time

            # remember the errors, update best net, pretty print
            self._after_epoch_helper(train_err, val_err, val_acc, train_batches, val_batches, dt, epoch)

    def _after_epoch_helper(self, train_err, val_err, val_acc, train_batches, val_batches, dt, epoch):
        """
        convenience fuction ussed by both doTraining and doTraining_with_generators
        :param train_err:
        :param val_err:
        :param val_acc:
        :param train_batches:
        :param val_batches:
        :param dt:
        :param epoch:
        :return:
        """
        # normalize the error to batches
        tr_loss_normed = train_err / train_batches
        val_loss_normed = val_err / val_batches
        val_acc_normed = val_acc / val_batches * 100

        self._print_epoch_summary(tr_loss_normed, val_loss_normed, val_acc_normed, dt, epoch)

        # record the parametesr if new best network
        val_loss_best = all([val_loss_normed < _ for _ in self.valError])
        if val_loss_best or not self.best_network_params:
            # new best performer or no other present yet
            self.best_network_params = lasagne.layers.get_all_param_values(self.network)

        # record for later
        self.trainError.append(tr_loss_normed)
        self.valError.append(val_loss_normed)
        self.valAccuracy.append(val_acc_normed)

    def _print_epoch_summary(self, tr_loss_normed, val_loss_normed, val_acc_normed, dt, epoch ):
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
        tr_loss_str = "{}{:.4f}{}".format(ansi.CYAN if tr_loss_best else "",
                                          tr_loss_normed,
                                          ansi.ENDC if tr_loss_best else "")

        val_loss_best = all([val_loss_normed < _ for _ in self.valError])
        val_loss_str = "{}{:.4f}{}".format(ansi.GREEN if val_loss_best else "",
                                           val_loss_normed,
                                           ansi.ENDC if val_loss_best else "")

        val_acc_best = all([val_acc_normed > _ for _ in self.valAccuracy])
        val_acc_str = "{}{:.4f}{}".format(ansi.MAGENTA if val_acc_best else "",
                                          val_acc_normed,
                                          ansi.ENDC if val_acc_best else "")

        prettyString = "%d\t\t%s\t\t%s\t\t%s\t\t%.3f" % (epoch, tr_loss_str, val_loss_str, val_acc_str, dt)
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

    # def predictLabels()
    def plot_error(self):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.trainError, 'b-')
        plt.plot(self.valError, 'g-')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "val"])

        plt.subplot(2, 1, 2)
        plt.plot(self.valAccuracy, 'b-')
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
