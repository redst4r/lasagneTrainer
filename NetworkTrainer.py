import time
import matplotlib.pyplot as plt
from batchgenerators import iterate_minibatches
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

    def doTraining(self, trainData, trainLabels, valData, valLabels, train_fn, val_fn, pred_fn, epochs, batchsize):
        """
        train_fn, val_fn are theano.functions
        """

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

            prettyString = "%d\t\t%.4f\t\t%.3f\t\t%.3f\t\t%.3f" % (epoch, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100, dt)
            print(prettyString)
            # print("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, )
            # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

            # record for later
            self.trainError.append(train_err / train_batches)
            self.valError.append(val_err / val_batches)
            self.valAccuracy.append(val_acc / val_batches * 100)
            # W_array.append(l_hid1.W.get_value().copy())
            # W_array.append(lConv1.W.get_value().copy())

        # finally, predict the labels
        # predLabels = np.argmax(pred_fn(valData), axis=1)  # returns a sample x 10 array, then take the label with the highest score

        # return predLabels
        self.plot_error()

    # def predictLabels()
    def plot_error(self):
        plt.figure()
        plt.plot(self.trainError, 'b-')
        plt.plot(self.valError, 'g-')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "val"])
