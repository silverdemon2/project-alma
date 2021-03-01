from ._template import TemplateEstimator
from ._template import TemplateClassifier
from ._template import TemplateTransformer
from .sampling_bayesian_encoder import SamplingBayesianEncoder

from ._version import __version__

__all__ = ['TemplateEstimator', 'TemplateClassifier', 'TemplateTransformer', 'SamplingBayesianEncoder',
           '__version__']
