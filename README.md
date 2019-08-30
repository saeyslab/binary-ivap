# IVAP defense against adversarial examples

This repository contains a reference implementation for our defense against [adversarial examples](https://adversarial-ml-tutorial.org/introduction/), which uses a technique from conformal prediction called *inductive Venn-ABERS predictors* (IVAPs; see [1, 2, 3] for more details). This work has been published in the proceedings of the ESANN 2019 conference [4].

## Prerequisites

In order to run our code, it is necessary to install the following dependencies:

* Python 3.5.2 or higher (https://www.python.org/);
* TensorFlow 1.8.0 (https://www.tensorflow.org/);
* Keras 2.2.0 (https://www.keras.io/);
* NumPy 1.14.5 (http://www.numpy.org/);
* Foolbox 1.4.0 (https://foolbox.readthedocs.io).

## Running the code

There is a [Jupyter Notebook](https://jupyter.org/) `cifar10_demo.ipynb` in this repository which contains detailed explanations and demonstrations of how to use the IVAP defense on a subset of the CIFAR-10 data set.

## References

1. Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale probabilistic predictors with and without guarantees of validity." Advances in Neural Information Processing Systems. 2015. [PDF](https://papers.nips.cc/paper/5805-large-scale-probabilistic-predictors-with-and-without-guarantees-of-validity.pdf)
2. Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.
3. Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction." Journal of Machine Learning Research 9.Mar (2008): 371-421. [PDF](http://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)
4. Peck, Jonathan, Bart Goossens and Yvan Saeys. "Detecting adversarial examples with inductive Venn-ABERS predictors." European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. 2019. [PDF](https://biblio.ugent.be/publication/8622378/file/8622388.pdf)
