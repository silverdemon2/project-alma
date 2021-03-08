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

Introduction to Sampling Bayesian Encoder
============================================================

Target encoding is a simple and quick encoding that doesnt add to the dimensionality of the dataset.

Each value of the categorical variable is mapped to a target mean conditional given the value of the variable.

Target encoding is dependent on the distribution of the target which means target encoding requires careful validation as it can be prone to overfitting.

Target encoding fails to extract information from intra-category target variable distribution apart from its mean.



The main motivation of Bayesian target encoding is to use iner-category variance in addition to the target mean in encoding categorical variables.

Bayesian target encoding selects a conjugate prior for the conditional distribution of the target variable given the value of the categorical variable and updates it with the training examples to obtain a posterior distribution.
In the encoding layer each category is encoded using the first moments of the posterior distribution.

Bayesian target encoding can be represented as a hierarchical model that uses weak learners to discover new features.

In sampling bayesian target encoding we sample the posterior distribtion instead of taking expectations of its first moments.

Source:

- Michael Larionov. Sampling Techniques in Bayesian Target Encoding, 2020

- https://towardsdatascience.com/target-encoding-and-bayesian-target-encoding-5c6a6c58ae8c

- https://mattmotoki.github.io/beta-target-encoding.html

Sampling Bayesian Encoder implementation
============================================================

https://github.com/mlarionov/categorical-encoding/tree/sampling

This repo implements sampling bayesian target encoding techniques:

An `EncoderWrapper` is a sampling encoder and a prediction model (Random forest, SVM...).

In the fit method of the encoder the member `_accumulator` finds a prior distribution with target statistics for the entire training dataset (scales it down with a parameter)
and then updates a conditional posterior ditribution using conjugate priors for each value of each categorical variable

The conjugate prior is beta distribution for a binary classficiation task and normal gamma distribution for a regression task

The member `_accumulator` computes the parameters of the posterior distribution and smaples

The `transform` method of the encoder generates an augmented set of the training set with all categorical features encoded using samples from the posterior distribution

The prediction model of the `EncoderWrapper` is trained om the augmented set and we average the results of the models with different encoded values to predict.

I did minor modifications in the code to pass unit tests of the scikit template (check_estimator).

I have added the tag `no validation` because the specific unit tests in `test_sampling.py` use dataset with float('nan') and float('inf'),

2 estimator unit tests are still failing:

- check_methods_subset_invariance
- check_fit_idempotent

The sampling encoder should be used with an `EncoderWraper. Therefore we should ensure 'EncoderWrapper' pass the `check_estimator` tests for the predictions models we use.

The execution of the methods `fit`, `transform` and `fit_transform` is slow for this implementation compared to other category encoders.
One of the reason could be the augmentation of the dataset. However this implementation should be optimized.
For example the `fit` method calls `transform` to update the set of features and the set of invariant columns.



Experiments
============================================================

- https://github.com/mlarionov/sampling_bayesian_encoder

 I was not able to run these experiments with my computer or with google-collab (memory issue problem, execution too slow...).

The file `adult_classif` corresponds to one of these experiments. The corresponding dataset is available here: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data


- https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark


This notebook benchmarks different categorical encoders on the dataset of the kaggle https://www.kaggle.com/c/cat-in-the-dat

The dataset of the competition contains categorical features with different modalities.

The code has been modified to include the sampling bayesian encoding in the benchmark and to fit the transformer on each training subset of the cross-validation (and not on the whole training subset).

To run this experiment, run the file `benchmark.py` (or execute the method run_cat_in_the_cat_xp)

The sampling encoder outperforms the other encoders on this experiment.

According to the article the encoder should have good performance on prediction task with categorical and numerical features.

https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8

Denis Vorotyntsev has worked on benchmarking categorical encoders with different type of validation.

Our previous experiment correspond to the single validation.

Double validation takes time more and sampling already induce a noise.


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