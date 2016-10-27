# VGG19 pretrained model

VGG-19, 19-layer model from the paper:
```
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/
```

Download pretrained weights from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
and put it into this very folder!

To instantiate this pretrained network (with pretrained weights)
```python
from pretrained.vgg19 import build_pretrained_vgg19
net = build_pretrained_vgg19()
```

To just create the architecture
```python
from pretrained.vgg19 import build_model
net = build_model()
```

# Googlenet
