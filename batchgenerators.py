import queue
import threading
import numpy as np
import time
import collections


def batch(iterable, batchsize):
    """
    splits the iterable into a couple of chunks of size n
    handy for iterating over batches
    :param iterable:  iterable to be chunked/batched
    :param batchsize: batchSize
    :return: gnerator over iterables
    """
    assert isinstance(iterable, collections.Iterable)  # TODO this does not guard against np.arrays as they are also iterable (over single elements)
    l = len(iterable)
    for ndx in range(0, l, batchsize):
        yield iterable[ndx:min(ndx + batchsize, l)]


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

    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)  # TODO untested

    for excerpt in batch(indices, batchsize=batchsize):  # chucks the indices into smaller batches, called excerpt
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

    indices = np.arange(nSamples)
    for excerpt in batch(indices, batchsize=batchsize):
        yield X_mmapped[excerpt], y_mmapped[excerpt].astype('int32') # TODO hack with the int32. should be handle when creating the labels. THis also breaks the unit test!


def random_crops_iterator(generator, cropSize):
    """
    for a given batchgenerator, apply random cropping to each element of the batches
    batchsize is conserved here
    :param generator: prooviding minibatches of images
    :param cropSize: will produce random crops of size:  (cropSize x cropSize)
    :return:
    """
    for batch_X, batch_label in generator:
        cropped_batch_X = []
        cropped_batch_Y = []
        for i in range(len(batch_X)):  # iterate over the samples in the batch
            img = batch_X[i]
            label = batch_label[i]

            img_cropped = _random_crop(img, cropSize)
            cropped_batch_X.append(img_cropped)
            cropped_batch_Y.append(label)

        # stack together into a 4D image
        stacked_batch = np.stack(cropped_batch_X)
        stacked_y  = np.array(cropped_batch_Y)

        assert stacked_batch.shape[:2] == batch_X.shape[:2]
        assert stacked_y.shape == batch_label.shape

        yield stacked_batch, stacked_y


def _random_crop(img, cropsize):
    "just does a random crop of the image. makes sure that we stay away from the image borders"
    channel, H, W = img.shape
    upper_left_x = np.random.randint(0, H - cropsize)
    upper_left_y = np.random.randint(0, W - cropsize)
    return img[:, upper_left_x:upper_left_x + cropsize, upper_left_y:upper_left_y + cropsize]