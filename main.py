import numpy as np
from data import a_test2, b_test2
from gaussian_algorithm import gaussian
from norms import residuals

x = gaussian(a_test2, b_test2)
residuals(a_test2, b_test2, x)
