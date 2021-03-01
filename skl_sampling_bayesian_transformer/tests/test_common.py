import pytest

from sklearn.utils.estimator_checks import check_estimator
from skl_sampling_bayesian_transformer import SamplingBayesianEncoder


@pytest.mark.parametrize(
    "Estimator", [SamplingBayesianEncoder]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
