import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Math
import MDAnalysis as mda
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.stats as stats
import re
import yaml         # Config file reader



# Import cycler
from cycler import cycler
import os

#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


# Set style of plt to times new roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.title_fontsize"] = 14
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"
# Make the ticksizes larger
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 2
# colors = ["f4a261","e76f51","5c5d8d","a8c69f","cce2a3"]

# light_blue_color = ["#1CADE4", "#42BA97", "#2683C6", "#3E8853", "#6EAC1C"]
# # make a custom color cycle
# custom_cycler = cycler(color=light_blue_color)
#plt.rcParams['axes.prop_cycle'] = custom_cycler