# lasagneTrainer
Some code around training a deep neural network created with the [lasagne framework](https://github.com/Lasagne/Lasagne)
Basically, one has to define the network, theano-functions for training and iterators that provide the data.

## First steps
Easiest access is the NetworkTrainer.doTraining() function which works without iterators
```python

trainer = NetworkTrainer(some_lasagne_net)
trainer.doTraining(Xtrain, Ytrain, Xval, Yval, train_function, val_function, epochs, batchsize)
trainer.plot_error()
```

Xtrain, Ytrain, Xval, Yval are just np.ndarrays of samples/labels. train_function, val_function are the functions that calculate training/validation error and update parameters (see DNN_constfunctions.py for a simple cross entropy example)

## The iterator interface
A lot more sophisticated things can be done when data is provided via iterators/generators (e.g. on the fly preprocessing, such as rotating).

1. Create two functions (one for training, one for validation), which take two input arguments and create a generator which returns tuples (X,y). 
```python
n = 50
X = np.random.rand(n,3,100,100)  # a couple of 100x100 RGB images
y = np.random.rand(n)>0.5  # some labels

def generator_factory(ix_x, ix_y):
    yield X[ix_x], y[ix_y]
```

2. Call NetworkTrainer.do_training_with_generators()
```python
train_ix = np.arange(40)
val_ix = np.arange(40,50)
trainer.do_training_with_generators(generator_factory, generator_factory, trainInput=train_ix, valInput=val_ix)
```
In each epoch, generator_factory is called with the sample indices (train_ix while training) and iterates over all the samples contained within

## TODO
- add a concrete, workable example
