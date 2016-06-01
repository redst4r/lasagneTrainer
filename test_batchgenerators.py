from batchgenerators import *
import numpy as np
import tempfile

def get_samples_labels(nSamples=100, nFeatures=10):
    X = np.random.rand(nSamples,nFeatures)
    y = np.random.rand(nSamples)
    return  X,y

def assert_equal_stacked_iterator_and_original(generator, originalX, originalY):
    "tests if the concatenated content of the iterator and the original input are the same"
    # exhaust the generator
    result_X, result_y = zip(*list(generator))

    #vstack X, hstack y (bit strange that its the other way around); for concatenate its actually consistent!

    result_X = np.concatenate(result_X)
    result_y = np.concatenate(result_y)

    assert np.all(result_X == originalX), 'generator and X differ'
    assert np.all(result_y == originalY), 'generator and y differ'

def test_iterate_batches_from_disk():
    "test if the concatenated minibatches of a mem-map array correspond to the original input"
    # create two temporary numpy arrays on disk
    fhX, fnameX = tempfile.mkstemp(suffix='.npy')
    fhY, fnameY = tempfile.mkstemp(suffix='.npy')

    X,y = get_samples_labels(100)

    np.save(fnameX, X)
    np.save(fnameY, y)

    gen = iterate_batches_from_disk(fnameX, fnameY, batchsize=3)

    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_iterate_minibatches():
    "does the concat of minibatches represent the original data. esp if the datasize is not a multiple of batchsize "
    batchsize = 30
    X,y = get_samples_labels(100)
    gen = iterate_minibatches(X, y, batchsize, shuffle=False)
    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_threaded_generator():

    X,y = get_samples_labels(100)

    # the 'expensive' load/preprocess generator that loads data
    expensive_gen = expensive_minibatch_iterator(X, y, expensive_time=0.05, batchsize=30, verbose=True)

    # wrap this in the threaded thingy so that loading is done in a seperate thread
    async_gen = threaded_generator(expensive_gen, num_cached=3)

    assert_equal_stacked_iterator_and_original(async_gen, X, y)