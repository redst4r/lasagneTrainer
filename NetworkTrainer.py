import numpy as np
import time
import matplotlib.pyplot as plt
import queue
import threading


# ------------------------------------------------------------------
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
# ------------------------------------------------------------------

def threaded_generator(generator, num_cached=5):
    """
    a asyncronous loader. num_cached determines how many minibatches get loading into mem at the same time

    heres how it works:
    - theres a queue of batch-contents (i.e. samples); each element in the queue is one minibatch
    - a seperate thread loads these batches into the queue (while the GPU is doing some calc)
        -> to load into the queue, we just poll a new element from `generator`

    - if the queue is full, the put() methods blocks, preventing further loading (waiting until new room in the queue is freed)
    - the main thread removes items from the queue via get() and hands them to the DNN for calculation

    :param generator:
    :param num_cached:
    :return:
    """
    the_queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:  # this is what loads the data!
            the_queue.put(item)
            print('loaded a batch')
        the_queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = the_queue.get()
    while item is not sentinel:
        print('getting a batch')
        yield item
        the_queue.task_done()
        item = the_queue.get()


def testing_async_loader():

    X = np.arange(500).reshape(100,5)
    Y = np.zeros(100)

    batch_thread = threaded_generator(iterate_minibatches(X,Y, 10))
    for a,b in batch_thread:

        print('-------------going to sleep')
        time.sleep(5)
        print('-------------waking up')


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

            print("Epoch\t\tTrLoss\t\tValLoss\t\tvalAcc\t\tTime")
            print(epoch, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100, sep='\t\t')
            # print("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
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
