#
# This program serves to exemplify the use of the class Functions for the configuration of the fuzzy functions
#

from fuzzy import Functions as fuzzyFunctions
import matplotlib.pyplot as plt
import numpy as np

#
# Define the range "x" to plot the fuzzy function in a range [0,10]
#
x = np.arange(0.0, 10, 0.1)

#
# Define the subplots where is going to show the functions
#
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

# Triangular function example
triangular = fuzzyFunctions.TriangularFunction(a=3, b=7, c=5, step=0.1)

ax1.plot(x, triangular.get_fuzzy_function())
ax1.set_title("Triangular function")

# Trapezoidal function example
trapezoidal = fuzzyFunctions.TrapezoidalFunction(a=0, b=9, c=3, d=6, step=0.1)

ax2.plot(x, trapezoidal.get_fuzzy_function())
ax2.set_title("Trapezoidal function")

# Sigmoid function example
sigmoid = fuzzyFunctions.SigmoidFunction(a=4, alpha=2, step=0.1)

ax3.plot(x, sigmoid.get_fuzzy_function())
ax3.set_title("Sigmoid function")

# Gaussian function example
gaussian = fuzzyFunctions.GaussianFunction(c=5, alpha=2, step=0.1)

ax4.plot(x,gaussian.get_fuzzy_function())
ax4.set_title("Gaussian function")

# Generalized bell function example
bell = fuzzyFunctions.GeneralizedBellCurveFunction(a=2, b=4, c=6, step=0.1)

ax5.plot(x, bell.get_fuzzy_function())
ax5.set_title("Generalized Bell Curve function")

# Tangent function example
tangent = fuzzyFunctions.TangentFunction(step=0.1)

ax6.plot(x, tangent.get_fuzzy_function())
ax6.set_title("Tangent function")

plt.tight_layout()
plt.show()
