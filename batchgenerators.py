try:
    import queue
except ImportError:  # for python2.7
    import Queue as queue

import threading
import numpy as np
import time
import collections


def exhaust_generator(gen):
    "just get all elements from the generator and merge the batches"
    X, y = zip(*list(gen))
    return np.concatenate(X), np.concatenate(y)


def batch(some_iterable, batchsize):
    """
    splits the iterable into a couple of chunks of size n
    handy for iterating over batches
    :param some_iterable:  iterable to be chunked/batched
    :param batchsize: batchSize
    :return: gnerator over iterables
    """
    assert isinstance(some_iterable, collections.Iterable)  # TODO this does not guard against np.arrays as they are also iterable (over single elements)
    l = len(some_iterable)
    for ndx in range(0, l, batchsize):
        yield some_iterable[ndx:min(ndx + batchsize, l)]


def batch_stacker(the_gen, batchsize):
    """
    takes a generator (yielding X,y tuples) and creates batches out of it, i.e. a batch of 10 3x28x28 images will
    yield a 10x3x28x28 np.array
    :param the_gen:
    :param batchsize:
    :return:
    """
    counter = 0
    currentX, currentY = [], []

    for A, B in the_gen:
        currentX.append(A)
        currentY.append(B)
        counter += 1

        if counter == batchsize:
            stackedX = np.stack(currentX)
            stackedY = np.stack(currentY)
            yield stackedX, stackedY

            counter = 0
            currentX, currentY = [], []

    # emit the lasta batch if still some data
    if currentX != [] and currentY != []:
        stackedX = np.stack(currentX)
        stackedY = np.stack(currentY)
        yield stackedX, stackedY


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    standard mini batch iterator over a dataset X,y, optional shuffling
    inputs,targets are just numpy.arrays
    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """
    assert len(inputs) == len(targets)

    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)

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
    for x_batch, y_batch in iterate_minibatches(X, y, batchsize, shuffle=False):
        if verbose:
            print('\t\tdoing expensive loading')
        time.sleep(expensive_time)  # just emulate some expensive loading/preprocess
        if verbose:
            print('\t\tdone expensive loading')
        yield x_batch, y_batch


def iterator_zscore_from_whole_data(batchgen, dataMean, dataStd):
    """
    gets the batches from the generator, subtract mean and std (not of the batch itself,
    but the given ones, calc over the entire data) and returns zscored batches
    :param batchgen:
    :param dataMean:
    :param dataStd:
    :return:
    """
    for Xbatch, yBatch in batchgen:
        assert isinstance(Xbatch, np.ndarray), 'the batch has to an np.array; instead %s' %type(Xbatch)
        assert Xbatch.shape[1:] == dataMean.shape == dataStd.shape
        Xbatch_zscored = (Xbatch - dataMean) / dataStd
        yield Xbatch_zscored, yBatch


def threaded_generator(generator, num_cached=5, verbose=False):
    """
    a asyncronous loader. num_cached determines how many minibatches get loading into mem at the same time

    heres how it works:
    - theres a queue of batch-contents (i.e. samples); each element in the queue is one minibatch
    - a seperate thread loads these batches into the queue (while the GPU is doing some calc)
        -> to load into the queue, we just poll a new element from `generator`

    - if the queue is full, the put() methods blocks, preventing further loading (waiting until new room in the queue is freed)
    - the main thread removes items from the queue via get() and hands them to the DNN for calculation

    :param generator:
    :param num_cached: how many batches should maximally be loaded into memory at the same time (use small values if Mem issues appear)
    :param verbose:
    :return:
    """
    the_queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        if verbose:
            dt = time.time()  # measure how long the loading takes
        for item in generator:  # this is what loads the data!
            the_queue.put(item)
            if verbose:  #TODO this should go before .put() in case the queue is full and put() blocks
                elapseTime = time.time() - dt
                print('loaded a batch in %.2f sec into queue [%d/%d]' % (elapseTime, the_queue.qsize(), the_queue.maxsize))
                dt = time.time()
        the_queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = the_queue.get()   # TODO for vrebose: the first consumption is not reported
    while item is not sentinel:
        if verbose:
            dt_consume = time.time()
        yield item
        the_queue.task_done()
        if verbose:
            dt_consume = time.time()-dt_consume
            print("consuming from queue in %.2f sec [%d/%d]" % (dt_consume,the_queue.qsize(), the_queue.maxsize))

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

    #TODO SHUFFLING?!
    indices = np.arange(nSamples)
    for excerpt in batch(indices, batchsize=batchsize):
        yield X_mmapped[excerpt], y_mmapped[excerpt]


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
    assert img.ndim == 3, "dimension is wrong, should be 3D: C x W x H"
    channel, H, W = img.shape
    upper_left_x = np.random.randint(0, H - cropsize)
    upper_left_y = np.random.randint(0, W - cropsize)
    return img[:, upper_left_x:upper_left_x + cropsize, upper_left_y:upper_left_y + cropsize]


def flip_rotate_iterator(generator):
    "randomly applies horz/vertical flips and 90/180/270 degree rotations to each element of hte generator"
    for batch_X, batch_label in generator:
        new_batch_X = []
        new_batch_Y = []
        for i in range(len(batch_X)):  # iterate over the samples in the batch
            img = batch_X[i]
            label = batch_label[i]

            img_flip_rot = _random_flip_rotate(img)
            new_batch_X.append(img_flip_rot)
            new_batch_Y.append(label)

        # stack together into a 4D image
        stacked_batch = np.stack(new_batch_X)
        stacked_y  = np.array(new_batch_Y)

        assert stacked_batch.shape[:2] == batch_X.shape[:2]
        assert stacked_y.shape == batch_label.shape

        yield stacked_batch, stacked_y


def _random_flip_rotate(img):
    assert len(img.shape) == 3, 'channels x width x height required'
    flipH = np.random.rand() > 0.5
    flipV = np.random.rand() > 0.5

    rotate = np.random.choice([0,90,180,270])

    def fH(X): return X[:,:,::-1]

    def fV(X): return X[:,::-1,:]

    if flipH:
        img = fH(img)
    if flipV:
        img = fV(img)

    if rotate == 90:
        img = fH(img).transpose([0,2,1])  # 90rotation is just flip + transpose
    elif rotate==180:
        img = fV(fH(img))  # 180 is flip in both directions
    elif rotate==270:
        img = fV(img).transpose([0, 2, 1])
    else:
        pass

    return img