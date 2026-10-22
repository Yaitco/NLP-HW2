# Импорты во всём датасете TinySMS

Проверен исходный train: **11 838 003** примеров, до числовой очистки и фильтра длины. Отдельно посчитан текущий train: **10 609 815** примеров. Исходный test содержит 1 919 строк без эталонного кода, поэтому набор импортов для него неизвестен.

Обычные import/from: исходный train — **269 223 (2.2742%)**, текущий train — **124 631 (1.1747%)**.

Динамические импорты: 14 примеров в исходном train и 10 в текущем; во всех случаях импортируется math. Примеров только с динамическим импортом: 14 и 10 соответственно.

Всего с обычным или динамическим импортом: **269 237** в исходном train и **124 641** в текущем.

## Библиотеки с объединением подмодулей

Число примеров с библиотекой, включая динамические импорты. Повторения одной библиотеки внутри примера считаются один раз. Категории пересекаются.

| Библиотека | Исходный train | Текущий train |
|---|---:|---:|
| `math` | 261 624 | 121 756 |
| `sympy` | 9 955 | 2 130 |
| `fractions` | 1 561 | 1 198 |
| `scipy` | 154 | 74 |
| `decimal` | 126 | 91 |
| `numpy` | 126 | 48 |
| `itertools` | 54 | 28 |
| `datetime` | 45 | 20 |
| `random` | 24 | 5 |
| `statistics` | 13 | 10 |
| `operator` | 3 | 3 |
| `functools` | 2 | 2 |
| `collections` | 2 | 1 |
| `calendar` | 1 | 1 |
| `cmath` | 1 | 0 |

## Модули в обычных import/from

| Модуль | Исходный train | Текущий train |
|---|---:|---:|
| `math` | 261 610 | 121 746 |
| `sympy` | 9 953 | 2 129 |
| `fractions` | 1 561 | 1 198 |
| `sympy.solvers` | 269 | 108 |
| `decimal` | 126 | 91 |
| `numpy` | 122 | 46 |
| `scipy.optimize` | 80 | 13 |
| `itertools` | 54 | 28 |
| `datetime` | 45 | 20 |
| `scipy.stats` | 38 | 33 |
| `scipy.special` | 36 | 28 |
| `random` | 24 | 5 |
| `statistics` | 13 | 10 |
| `numpy.linalg` | 4 | 1 |
| `sympy.abc` | 3 | 1 |
| `operator` | 3 | 3 |
| `sympy.physics.units` | 2 | 1 |
| `functools` | 2 | 2 |
| `collections` | 2 | 1 |
| `numpy.polynomial` | 1 | 1 |
| `calendar` | 1 | 1 |
| `cmath` | 1 | 0 |
| `scipy.misc` | 1 | 0 |

## Формы обычных импортов

Составные инструкции разбиты по импортируемым именам. Например, from math import sqrt, pi учитывается как две формы, но как один пример с math.

| Форма | Исходный train | Текущий train |
|---|---:|---:|
| `import math` | 259 641 | 120 164 |
| `import sympy` | 7 722 | 1 186 |
| `from fractions import Fraction` | 1 551 | 1 198 |
| `from sympy import solve` | 1 048 | 636 |
| `import sympy as sp` | 877 | 168 |
| `from sympy import symbols` | 680 | 345 |
| `from sympy import Symbol` | 649 | 412 |
| `from math import sqrt` | 504 | 385 |
| `from math import ceil` | 450 | 368 |
| `from math import pi` | 414 | 397 |
| `from sympy.solvers import solve` | 269 | 108 |
| `from math import comb` | 236 | 210 |
| `from math import gcd` | 216 | 110 |
| `from decimal import Decimal` | 126 | 91 |
| `import numpy as np` | 121 | 46 |
| `from sympy import Eq` | 66 | 29 |
| `from math import floor` | 53 | 37 |
| `from scipy.optimize import linprog` | 52 | 2 |
| `from math import factorial` | 44 | 30 |
| `from math import radians` | 35 | 34 |
| `from itertools import combinations` | 35 | 21 |
| `from math import sin` | 29 | 29 |
| `from scipy.stats import binom` | 26 | 24 |
| `from datetime import datetime` | 21 | 4 |
| `from datetime import date` | 20 | 14 |
| `from scipy.optimize import fsolve` | 20 | 10 |
| `from scipy.special import comb` | 20 | 14 |
| `from itertools import product` | 19 | 4 |
| `from datetime import timedelta` | 19 | 5 |
| `from math import pow` | 18 | 15 |
| `from scipy.special import binom` | 16 | 14 |
| `from math import tan` | 16 | 14 |
| `from math import degrees` | 13 | 10 |
| `from math import atan` | 12 | 11 |
| `from random import randint` | 12 | 4 |
| `from scipy.stats import norm` | 11 | 9 |
| `from sympy import lcm` | 11 | 11 |
| `from fractions import gcd` | 10 | 0 |
| `from statistics import mean` | 9 | 9 |
| `from sympy import pi` | 9 | 6 |
| `from random import choice` | 9 | 0 |
| `from scipy.optimize import minimize` | 7 | 1 |
| `from math import isclose` | 6 | 1 |
| `from decimal import getcontext` | 6 | 6 |
| `from math import cos` | 5 | 5 |
| `from sympy import Rational` | 5 | 3 |
| `from math import comb as choose` | 4 | 4 |
| `from math import exp` | 4 | 3 |
| `from numpy.linalg import solve` | 4 | 1 |
| `import random` | 4 | 1 |
| `from math import factorial as fact` | 3 | 3 |
| `from sympy import ceiling` | 3 | 3 |
| `from statistics import NormalDist` | 3 | 0 |
| `from itertools import permutations` | 3 | 3 |
| `from math import perm` | 3 | 3 |
| `from sympy import solveset` | 2 | 2 |
| `from sympy import diff` | 2 | 1 |
| `from sympy.abc import x` | 2 | 0 |
| `from sympy import sqrt` | 2 | 2 |
| `from random import random` | 2 | 0 |
| `from sympy import Derivative` | 2 | 0 |
| `from datetime import time` | 2 | 0 |
| `from functools import reduce` | 2 | 2 |
| `from collections import Counter` | 2 | 1 |
| `from operator import truediv` | 2 | 2 |
| `import sympy as sym` | 2 | 0 |
| `from sympy import FiniteSet` | 1 | 1 |
| `from math import acos` | 1 | 0 |
| `from sympy.physics.units import feet` | 1 | 1 |
| `from sympy.physics.units import ft` | 1 | 1 |
| `from sympy.abc import y` | 1 | 0 |
| `from math import log2` | 1 | 1 |
| `from scipy.optimize import minimize_scalar` | 1 | 0 |
| `from sympy.abc import pi` | 1 | 1 |
| `from math import erf` | 1 | 0 |
| `from scipy.stats import hypergeom` | 1 | 0 |
| `from math import log` | 1 | 0 |
| `from math import hypot` | 1 | 0 |
| `from numpy import array` | 1 | 0 |
| `from numpy.polynomial import Polynomial as P` | 1 | 1 |
| `from calendar import monthrange` | 1 | 1 |
| `from sympy.physics.units import foot` | 1 | 0 |
| `from sympy.physics.units import yard` | 1 | 0 |
| `from operator import mul` | 1 | 1 |
| `from cmath import sqrt` | 1 | 0 |
| `from statistics import variance` | 1 | 1 |
| `from scipy.misc import derivative` | 1 | 0 |
| `import datetime` | 1 | 0 |

Разбор выполнен через AST без запуска решений. Наличие импорта не доказывает использование импортированного имени при вычислении ответа. Упоминания в строках и комментариях исключены. Среди кандидатов с подстрокой import нет ошибок разбора AST.

Время аудита: 14.9 с. Только CPU, низкий приоритет.

[Полная статистика JSON](imports_full_dataset.json); [анализатор](../artifacts/imports_audit/analyze.py).
