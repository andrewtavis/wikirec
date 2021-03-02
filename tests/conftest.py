"""
Fixtures
--------
"""

import numpy as np

import pytest

import wikirec

from wikirec import autoencoder
from wikirec import data
from wikirec import model
from wikirec import topic_model
from wikirec import utils

np.random.seed(42)


@pytest.fixture(params=[])
def fixture(request):
    return request.param
