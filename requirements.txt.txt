Some of the Commands are below

pip install fpdf
pip install pandas==1.3.3
pip install numpy==1.21.2
pip install matplotlib==3.4.3
pip install seaborn==0.11.2
pip install scikit-learn==0.24.2

Below are the import statements 

# Basic Data Manipulation Libraries
import pandas as pd
import numpy as np

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Model Evaluation and Processing Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Utility Libraries
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# Suppress Warnings
warnings.filterwarnings('ignore')

# Inline plotting for Jupyter notebooks
%matplotlib inline

These imports cover the necessary libraries for data manipulation, visualization, and handling warnings.
If error occur while installation, please do let me know