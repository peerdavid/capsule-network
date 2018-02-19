# Capsule Network
Kears implementation of CapsuleNet [1] based on the already existing implementations [2] and [3]. 

## Note
* main.py can be used for both, training and testing. During tests call it via -t and -w to set the weights file
* Test augmentation parameters such as rotation, shift etc. can be set in the test_generator (currently its not a cmd arg)


## Differences to [1]
* Added epsilon to margin_loss to avoid numerical errors during forward pass
* All parameters of [1] are set as default. Some of those can be changed as cmd arg.

# References:
[[1]](https://arxiv.org/pdf/1710.09829.pdf) Sabour et al., Dynamic Routing Between Capsules, NIPS 2017 <br />
[[2]](https://github.com/XifengGuo/CapsNet-Keras/) XifengGuo/CapsNet-Keras <br />
[[3]](https://github.com/wballard/CapsNet-Keras/) wballard/CapsNet-Keras <br />
