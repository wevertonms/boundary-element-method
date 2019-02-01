"""Módulo de testes.
"""

import sys
import os

MYPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MYPATH + "/../")
import potential
import elastic


def test_potential_gauss():
    """Teste o método gauss() do módulo potential.
    """
    assert potential.Model.gauss(1) == (2, 0)


def test_elastic_gauss():
    """Teste o método gauss() do módulo elastic.
    """
    assert elastic.Model.gauss(1) == (2, 0)
