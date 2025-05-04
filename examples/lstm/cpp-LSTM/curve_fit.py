import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f3(x, x_1, x_3):
    return x * x_1 + (x ** 3) * x_3

def f5(x, x_1, x_3, x_5):
    return x * x_1 + (x ** 3) * x_3 + (x ** 5) * x_5

xdata = np.linspace(0, 2, 100)
ydata = np.tanh(xdata)

# print(xdata, ydata)

popt, pcov = curve_fit(f3, xdata, ydata)

print (popt, pcov)
