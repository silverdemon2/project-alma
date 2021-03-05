.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-template/badge/?version=latest
.. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

Introduction to sampling bayesian target encoder
============================================================

Target encoding is simple and quick encoding that doesnt add to the dimensionality of the dataset. Each value of the categorical variable is mapped to a target mean conditional given the value of the variable. Target encoding is dependent on the distribution of the target which means target encoding requires careful validation as it can be prone to overfitting. Target Encoding fails to extract information from intra-category target variable distribution apart from its mean.

The main motivation of Bayesian Target Encoding is to use iner-category variance in addition to the target mean in encoding categorical variables. In Bayesian target encoding we select a conjugate prior for the conditional distribution of the target variable given the value of the categorical variable and we update it with the training examples to obtain a posterior distribution. In the encoding layer each category is encoded using the first moments of the posterior distribution.

Bayesian target encoding can be represented as a hierarchical model that uses weak learners to discover new features. In sampling bayesian target encoding we sample the posterior distribtion instead of taking expectations of its first moments.

Source:

- Michael Larionov. Sampling Techniques in Bayesian Target Encoding, 2020.

- https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c

- https://mattmotoki.github.io/beta-target-encoding.html

SamplingBayesianEncoder implementation
============================================================

Sampling bayesian target encoding techniques have been implemented in https://github.com/mlarionov/categorical-encoding/tree/sampling

An `EncoderWrapper` is a sampling encoder and a model (Random forest, SVM...).

A sampling encoder has a member `_accumulator` corresponding to normal gamma distribution for regression or to beta distribution for binary classification

In the fit method of the encoder the accumulator finds a prior distribution with target statistics for the entire training dataset (scale it down with a parameter)
and then updates a conditional posterior ditribution using conjugate priors for each value of each categorical variable

The `transform` method of the encoder generates an augmented set of the training set with all categorical features encoded using samples from the posterior distribution

The model of the `EncoderWrapper` is trained om the augmented set and we average the K results of the models with K different encoded values to predict.

I did minor modifications in the code to pass unit tests of the scikit template (check_estimator)

I have added the tag `no validation` because the specific unit tests in `test_sampling.py` use dataset with float('nan') and float('inf'),

2 tests are still failing:

- check_methods_subset_invariance
- check_fit_idempotent

Sampling Bayesian Encoder should be used with an encoderwraper. We should check the EncoderWraper pass the check_estimator tests.

The methods `fit`, `transform` and `fit_transform` are slow for this implementation compared to other category encoders.
One of the reason could be the augmentation of the dataset. However this implementation should be optimized.
For example the `fit` method calls `transform` to update the set of features and the set of invariant columns.



Experiments
============================================================

- https://github.com/mlarionov/sampling_bayesian_encoder

 I was not able to run these experiments with my computer or with google-collab (memory issue, too slow...). The file adult_classif corresponds to one of these experiments. The corresponding dataset is available:


- https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark


This notebook benchmarks different categorical encoders on the dataset of the competition https://www.kaggle.com/c/cat-in-the-dat

The dataset of the competition contains categorical features with different modalities.

I've modified the code to include the sampling bayesian encoding in the benchmark and to fit the transformer on each training subset of the cross-validation (and not on the whole training subset)

To run this experiment run the file `benchmark.py` (or execute the method run_cat_in_the_cat_xp)

The sampling encoder outperforms the other encoders.

According to the article the encoder should have good performance on prediction task with categorical and numerical features.

https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8

Denis Vorotyntsev has worked on benchmarking categorical encoders.

Our previous experiment correspond to the single validation

Double validation takes time and sampling already induce a noise.


TODO
============================================================

- Test on credit data: some columns are numeric, some columns are categoric.

- Test cat_in_the_dat with specific columnn to make some conclusions on the performance for encoding category with low/high modality.

- Try to adapt and test Denis Vorotyntsev benchmark.

- Clean the code and improve the doc

Links
============================================================
Michael Larionov. Sampling Techniques in Bayesian Target Encoding, 2020.

https://github.com/scikit-learn-contrib/category_encoders/

https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c

https://github.com/aslakey/CBM_Encoding

https://www.kaggle.com/mmotoki/hierarchical-bayesian-target-encoding

https://mattmotoki.github.io/beta-target-encoding.html

https://github.com/mlarionov/sampling_bayesian_encoder

https://github.com/mlarionov/categorical-encoding/tree/sampling

https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark

https://github.com/DenisVorotyntsev/CategoricalEncodingBenchmark

https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8


https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c

https://mattmotoki.github.io/beta-target-encoding.html