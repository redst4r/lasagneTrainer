import queue
import threading
import numpy as np
import time

# ------------------------------------------------------------------
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    standard mini batch iterator over a dataset X,y, optional shuffling
    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """
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

    # get the last batch, which might be smaller than the other (if len(inputs) is not a multiple of batchsize)
    new_start = start_idx+batchsize
    if new_start<len(inputs):  # however, if it is not smaller, we have to make sure we dont return an empty batch
        if shuffle:
            excerpt = indices[new_start:]
        else:
            excerpt = slice(new_start, len(inputs))
        yield inputs[excerpt], targets[excerpt]

# ------------------------------------------------------------------

def expensive_minibatch_iterator(X,y,batchsize, expensive_time=5,verbose=False):
    """
    just a testcase for a scenario where loading/preprocessing a batch is very expensive
    can be used to test the threaded generator, which does expensive load/preprocess while the GPU is working
    :param X:
    :param y:
    :param batchsize:
    :return:
    """
    # TODO: utilize iterate_minibatches instead

    for start_idx in range(0, len(X) - batchsize + 1, batchsize):
        if verbose:
            print('\t\tdoing expensive loading')
        time.sleep(expensive_time)  # just emulate some expensive loading/preprocess

        if verbose:
            print('\t\tdone expensive loading')
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

    # last batch
    new_start = start_idx + batchsize
    if new_start < len(X):
        excerpt = slice(new_start, len(X))
        yield X[excerpt], y[excerpt]


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
            # print('loaded a batch')
        the_queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = the_queue.get()
    while item is not sentinel:
        # print('getting a batch')
        yield item
        the_queue.task_done()
        item = the_queue.get()


def async_loader_test():

    X = np.arange(500).reshape(100,5)
    Y = np.zeros(100)

    batch_thread = threaded_generator(iterate_minibatches(X,Y, 10))
    for a,b in batch_thread:

        print('-------------going to sleep')
        time.sleep(5)
        print('-------------waking up')


def iterate_batches_from_disk(X_npyfile, y_npyfile, batchsize):
    """
    iterates over X,y arrays stored on disk in minibatches without loading the whole array into mem
    :param X_npyfile: file where X array is stored (via np.save). IMPORTANT: doesnt work with np.savez()
    :param y_npyfile: ---"----
    :param batchsize: size of a single minibatch
    :return: generator over minibatches
    """
    X_mmapped = np.load(file=X_npyfile, mmap_mode='r')
    y_mmapped = np.load(file=y_npyfile, mmap_mode='r')

    nSamples = X_mmapped.shape[0]
    assert len(y_mmapped) == nSamples, "X and y have different number of samples (1st dimension)"
    for start_idx in range(0, nSamples - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X_mmapped[excerpt], y_mmapped[excerpt]

    # last batch
    new_start = start_idx + batchsize
    if new_start < nSamples:
        excerpt = slice(new_start, nSamples)
        yield X_mmapped[excerpt], y_mmapped[excerpt]