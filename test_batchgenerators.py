from batchgenerators import *
import numpy as np
import tempfile

"""
-----------------------------------------------------------------------------------------------------------------------
utility functions
-----------------------------------------------------------------------------------------------------------------------
"""

def create_disk_arrays(nSamples):
    # create two temporary numpy arrays on disk
    fhX, fnameX = tempfile.mkstemp(suffix='.npy')
    fhY, fnameY = tempfile.mkstemp(suffix='.npy')

    X,y = get_samples_labels(nSamples)

    np.save(fnameX, X)
    np.save(fnameY, y)
    return fnameX, fnameY, X,y


def assert_all_batches_nonempty(generator):
    " checks if any batch contains no items, that is an empty numpy array"
    batches = list(generator)

    batchsizes_X = [len(b[0]) for b in batches]
    batchsizes_Y = [len(b[1]) for b in batches]

    assert all(batchsizes_X), 'contains an empty minibatch'
    assert all(batchsizes_Y), 'contains an empty minibatch'


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

    assert result_X.dtype == originalX.dtype, 'generator and x have different dtype'
    assert result_y.dtype == originalY.dtype, 'generator and x have different dtype'
    assert np.all(result_X == originalX), 'generator and X differ'
    assert np.all(result_y == originalY), 'generator and y differ'

"""
-----------------------------------------------------------------------------------------------------------------------
iterate_batches_from_disk
-----------------------------------------------------------------------------------------------------------------------
"""

def test_iterate_batches_from_disk():
    "test if the concatenated minibatches of a mem-map array correspond to the original input"
    fnameX, fnameY, X, y = create_disk_arrays(nSamples=100)
    gen = iterate_batches_from_disk(fnameX, fnameY, batchsize=3)
    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_iterate_batches_from_disk_no_empty_batches():
    "test if the concatenated minibatches of a mem-map array correspond to the original input"

    fnameX, fnameY, _, _ = create_disk_arrays(nSamples=100)
    gen = iterate_batches_from_disk(fnameX, fnameY, batchsize=10)
    assert_all_batches_nonempty(gen)

"""
-----------------------------------------------------------------------------------------------------------------------
iterate_minibatches
-----------------------------------------------------------------------------------------------------------------------
"""
def test_iterate_minibatches():
    "does the concat of minibatches represent the original data. esp if the datasize is not a multiple of batchsize "
    batchsize = 30
    X,y = get_samples_labels(100)
    gen = iterate_minibatches(X, y, batchsize, shuffle=False)
    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_iterate_minibatches_no_empty_batches():
    "make sure that theres not empty batches (the last one might be a bit tricky e.g.). Esp problematic if samples is a mukltiple of batchsize"
    batchsize = 10
    X, y = get_samples_labels(100)
    gen = iterate_minibatches(X, y, batchsize, shuffle=False)
    assert_all_batches_nonempty(gen)


"""
-----------------------------------------------------------------------------------------------------------------------
threaded_generator
-----------------------------------------------------------------------------------------------------------------------
"""
def test_threaded_generator():

    X,y = get_samples_labels(100)
    # the 'expensive' load/preprocess generator that loads data
    expensive_gen = expensive_minibatch_iterator(X, y, expensive_time=0.05, batchsize=30, verbose=True)
    # wrap this in the threaded thingy so that loading is done in a seperate thread
    async_gen = threaded_generator(expensive_gen, num_cached=3)
    assert_equal_stacked_iterator_and_original(async_gen, X, y)


"""
-----------------------------------------------------------------------------------------------------------------------
cropping generator
-----------------------------------------------------------------------------------------------------------------------
"""
def create_4D_Data(N):
    X = np.random.rand(N,1,28,28)
    y = np.random.rand(N)
    return X, y
def test_random_crops_iterator():
    "assert that the debatched data is the same as the original"
    N = 100
    X,y =  create_4D_Data(N)

    cropSize = 10
    thegen = random_crops_iterator(iterate_minibatches(X, y, batchsize=30, shuffle=False), cropSize=cropSize)

    A, B = zip(*list(thegen))  # exhaust the generator

    X2 = np.vstack(A)
    y2 = np.hstack(B)

    # cannot compare X and X2 directly, just their sizes have to be consistent
    assert X2.shape == (N,X.shape[1], cropSize, cropSize)
    assert np.all(y==y2)


def test_random_crops_iterator_batchsize():
    "ensure that the batches have the requested number of elements"
    N = 100
    X, y = create_4D_Data(N)
    cropSize = 10
    batchsize = 30
    thegen = random_crops_iterator(iterate_minibatches(X, y, batchsize=30, shuffle=False), cropSize=cropSize)

    A,B = next(thegen)

    assert len(A) == batchsize and len(B)==batchsize

def test_random_crops_iterator_no_empty_batches():
    batchsize = 10
    X, y = create_4D_Data(N=100)
    gen = random_crops_iterator(iterate_minibatches(X, y, batchsize=batchsize, shuffle=False), cropSize=10)
    assert_all_batches_nonempty(gen)
