# BetaBinomial

![tests](https://github.com/muhammedhasan/betabinomial/actions/workflows/python-app.yml/badge.svg)
[![pypi](https://img.shields.io/pypi/v/betabinomial.svg)](https://pypi.python.org/pypi/betabinomial)
[![Documentation Status](https://readthedocs.org/projects/betabinomial/badge/?version=latest)](https://betabinomial.readthedocs.io/en/latest/?badge=latest)

Implementation of Beta-Binomial (https://en.wikipedia.org/wiki/Beta-binomial_distribution) in python for parameters inference with moment method estimation and statistical testing on count data.

[Documentation](https://betabinomial.readthedocs.io/en/latest/)

## Installation

```
pip install betabinomial
```

## Example

```python
import numpy as np
from betabinomial import BetaBinomial, pval_adj


bb = BetaBinomial()

# total counts
n = np.array([
  [5, 2, 5, 6, 6],
  [8, 8, 0, 9, 1],
  [8, 2, 6, 1, 7]
])
# event count
k = np.array([
  [3, 1, 4, 1, 2],
  [8, 7, 0, 9, 1],
  [0, 0, 0, 0, 2]
])

# Infer `alpha` and `beta` parameters from counts
bb.infer(k, n)

bb.alpha
# [[ 11.45811965]
#  [121.01628682]
#  [0.43620744]]

bb.beta
# [[13.332114  ]
#  [ 4.97492014]
#  [ 5.41047636]]

# Statistical testing with inferred `alpha` and `beta`
pval = bb.pval(k, n, alternative='two-sided')
# array([[0.33287737, 0.44653957, 0.06266123, 0.35378069, 0.85568061],
#        [0.        , 0.53825136, 0.        , 0.        , 0.        ],
#       [0.67209923, 0.26713023, 0.57287758, 0.14921533, 0.10535054]])

# Adjust p-value with multiple testing correction
padj = pval_adj(pval)
# array([[0.53067103, 0.60891759, 0.18798369, 0.53067103, 0.85568061],
#        [0.        , 0.6610126 , 0.        , 0.        , 0.        ],
#        [0.72010631, 0.50086919, 0.6610126 , 0.31974714, 0.26337634]])
```

## Citation

If you use this package in academic publication, please cite:
