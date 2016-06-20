from batchgenerators import *
import numpy as np
import tempfile
import toolz

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


def get_samples_labels(nSamples=100, nFeatures=10):
    X = np.random.rand(nSamples,nFeatures)
    y = np.random.rand(nSamples)
    return  X,y


def assert_equal_stacked_iterator_and_original(generator, originalX, originalY):
    "tests if the concatenated content of the iterator and the original input are the same"
    # exhaust the generator
    result_X, result_y = zip(*list(generator))

    result_X = np.concatenate(result_X)
    result_y = np.concatenate(result_y)

    assert result_X.dtype == originalX.dtype, 'generator and x have different dtype'
    assert result_y.dtype == originalY.dtype, 'generator and x have different dtype'
    assert np.all(result_X == originalX), 'generator and X differ'
    assert np.all(result_y == originalY), 'generator and y differ'

def assert_correct_batchsizes(generator, batchsize):
    "checks that the genearotrs produce the sbatchsize requested and the no batch is empty!"
    batches = list(generator)

    # check that no batch is None
    assert all([b[0] is not None for b in batches]), "x batch contains None"
    assert all([b[1] is not None for b in batches]), "y batch contains None"

    batchsizes_X = [len(b[0]) for b in batches]
    batchsizes_Y = [len(b[1]) for b in batches]

    "test empty batches"
    assert all(batchsizes_X), 'contains an empty minibatch'
    assert all(batchsizes_Y), 'contains an empty minibatch'

    "test requested batchsize"
    assert all([b == batchsize for b in  batchsizes_X[:-1] ]), "wrong batchsize"
    assert all([b == batchsize for b in  batchsizes_Y[:-1] ]), "wrong batchsize"


"""
-----------------------------------------------------------------------------------------------------------------------
iterate_batches_from_disk
-----------------------------------------------------------------------------------------------------------------------
"""

def test_iterate_batches_from_disk_faithful():
    "test if the concatenated minibatches of a mem-map array correspond to the original input"
    fnameX, fnameY, X, y = create_disk_arrays(nSamples=100)
    gen = iterate_batches_from_disk(fnameX, fnameY, batchsize=3)
    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_iterate_batches_from_disk_batchsize():
    "test if the concatenated minibatches of a mem-map array correspond to the original input"
    batchsize = 10
    fnameX, fnameY, _, _ = create_disk_arrays(nSamples=100)
    gen = iterate_batches_from_disk(fnameX, fnameY, batchsize=batchsize)
    assert_correct_batchsizes(gen, batchsize)

"""
-----------------------------------------------------------------------------------------------------------------------
iterate_minibatches
-----------------------------------------------------------------------------------------------------------------------
"""
def test_iterate_minibatches_faithful():
    "does the concat of minibatches represent the original data. esp if the datasize is not a multiple of batchsize "
    batchsize = 30
    X,y = get_samples_labels(100)
    gen = iterate_minibatches(X, y, batchsize, shuffle=False)
    assert_equal_stacked_iterator_and_original(gen, X, y)


def test_iterate_minibatches_batchsize():
    "make sure that theres not empty batches (the last one might be a bit tricky e.g.). Esp problematic if samples is a mukltiple of batchsize"
    batchsize = 10
    X, y = get_samples_labels(100)
    gen = iterate_minibatches(X, y, batchsize, shuffle=False)

    assert_correct_batchsizes(gen ,batchsize)


def test_iterate_minibatches_shuffle():

    batchsize = 10
    X, y = get_samples_labels(1000)

    gen = iterate_minibatches(X, y, batchsize, shuffle=True)

    result_X, result_y = zip(*list(gen))
    result_X = np.concatenate(result_X)
    result_y = np.concatenate(result_y)

    # check that they are NOT the same
    assert np.any(np.logical_not(np.isclose(result_X,X)))
    assert np.any(np.logical_not(np.isclose(result_y,y)))
    # if we're really unlucky, shuffle might preserve the same order
    # or shuffle y in such a way that 0/1 align even after shuffle
    # however, really unrealistic for 1000 samples
"""
-----------------------------------------------------------------------------------------------------------------------
threaded_generator
-----------------------------------------------------------------------------------------------------------------------
"""
def test_threaded_generator_faithful():

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


def test_random_crops_iterator_faithful():
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
    thegen = random_crops_iterator(iterate_minibatches(X, y, batchsize=batchsize, shuffle=False), cropSize=cropSize)
    assert_correct_batchsizes(thegen, batchsize)

"""
-----------------------------------------------------------------------------------------------------------------------
batch()
-----------------------------------------------------------------------------------------------------------------------
"""

def test_batch_faithful():
    "joining the batches must result the unbatched data"
    X = list(range(11))
    the_batches = batch(X, batchsize=3)
    X_debatched = toolz.reduce(lambda l1,l2: l1+l2, the_batches)

    assert X_debatched == X, 'different ouput prodcued'

def test_batch_batchsize():
    "bit tricky to use the other functionality for the generators, as those assume that the batchelements are tuples: (X,y)"
    batchsize = 3
    X = list(range(11))
    the_batches = list(batch(X, batchsize=batchsize))
    assert len(the_batches[0]) == batchsize, 'wrong batchsize produced'

def test_batch_nonempty():
    X = list(range(10))
    gen = batch(X, batchsize=2)
    the_batches = list(gen)
    assert all([_ is not None for _ in the_batches]) and all([_ != [] for _ in the_batches]), 'empty batch detected'


