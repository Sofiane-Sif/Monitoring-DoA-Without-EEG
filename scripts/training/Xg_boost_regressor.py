#!/usr/bin/env python
# coding: utf-8

# In[135]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_parquet('/Users/abhinavrajput/Desktop/PROJECTS/NoEEG_Borealis/doa-zero-eeg-sample_filtered/b1174f7e-0ca1-427b-999b-d52efb2a7c59.parquet', engine='pyarrow')

data = df

print(f"Shape of the data: {data.shape}")
print("Overview: \n")
data.head()


# In[136]:


import numpy as np
from scipy.interpolate import interp1d

def noise_interpolate(y, scale, noise_probability = 0.25):
    x_valid = np.where(~np.isnan(y))[0]
    y_valid = y[x_valid]  # Correctly extract y_valid from y using indices
    x_missing = np.where(np.isnan(y))[0]

    # Fit a quadratic spline on the valid data points
    quadratic_spline = interp1d(x_valid, y_valid, kind='quadratic', fill_value="extrapolate")

    # Interpolate the missing (NaN) values using the spline
    y[x_missing] = quadratic_spline(x_missing)

    noise_scale = scale*(np.max(y) - np.min(y))

    noise_mask = np.random.rand(*y.shape) < noise_probability


    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=noise_scale, size=y.shape)*noise_mask

    # Add noise to the polynomial values
    y_noisy = y + noise

    return y,y_noisy

bis = data['BIS'].values


# In[137]:


import copy

def generate_arrays(data):
    _,pnim= noise_interpolate(data['PNIm'].values.flatten(),0.05 )

    bis_ = data[['BIS']].values

    i = 0
    while(not (bis_[i] >= 0)):
        i = i+1

    dataf = copy.deepcopy(data)
    dataf.fillna(method='ffill', inplace=True)


    bis = data[['BIS']].values
    fc = dataf[['FC']].values
    spo2 = dataf[['SpO₂']].values
    co2 = dataf[['CO₂fe']].values

    bis = bis[i:]
    fc = fc[i:]
    spo2 = spo2[i:]
    co2 = co2[i:]
    pnim = pnim[i:]

    return bis,pnim,fc,spo2,co2



# In[138]:


bis,pnim,fc,spo2,co2 = generate_arrays(data)


import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))
plt.plot(bis[:11000], label="BIS", c="r")
plt.plot(fc[:11000], label="FC", c="g")
plt.plot(spo2[:11000], label="SPO2", c="b")
plt.plot(co2[:11000], label="CO2", c="orange")
plt.plot(pnim[:11000], label="PNIm")
#plt.plot(pnim[:11000], label="PNIm")

plt.xlabel("Time")
plt.ylabel("Time series values")
plt.title("All labels with BIS index")

plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# In[139]:


df = pd.DataFrame({'A': bis.flatten(), 'B': pnim.flatten(), 'C': co2.flatten(), 'D': spo2.flatten(), 'E': fc.flatten()})
df.describe()


# In[140]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # Add this import

import xgboost as xgb

def calcmse(bis, pnim, co2, spo2, fc, width=42, want_to_print = False, n_estim=4):
    df = pd.DataFrame({
        'A': bis.flatten(), 
        'B': pnim.flatten(), 
        'C': co2.flatten(), 
        'D': spo2.flatten(), 
        'E': fc.flatten()
    })

    window_size = width
    # Create lagged features for the desired window size
    for lag in range(1, window_size + 1):
        df[f'B_lag{lag}'] = df['B'].shift(lag)
        df[f'C_lag{lag}'] = df['C'].shift(lag)
        df[f'D_lag{lag}'] = df['D'].shift(lag)
        df[f'E_lag{lag}'] = df['E'].shift(lag)

    # Drop rows with NaN values resulting from the shift operation AFTER creating all lags
    df = df.dropna()

    # Define features (including all lags for B, C, D, and E) and target (A)
    features = [f'{var}_lag{lag}' for var in ['B', 'C', 'D', 'E',] for lag in range(1, window_size + 1)]
    X = df[features]
    y = df['A']

    n = 0.9
    # Calculate the split index
    split_index = int(n * len(X))

    # Split the data into training and testing sets
    X_train, X_ = X[:split_index], X[split_index:]
    y_train, y_ = y[:split_index], y[split_index:]

    n = 0.5

    split_index = int(n * len(X_))

    X_Val, X_test = X_[:split_index], X_[split_index:]
    y_Val, y_test = y_[:split_index], y_[split_index:]

    # Initialize the XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estim)

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)
    y_total = model.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    y_pred_val = model.predict(X_Val)
    mse_Val = mean_squared_error(y_Val, y_pred_val)


    if(want_to_print):
        print("TEST")
        plt.figure(figsize=(16,6))
        plt.plot(y_test.values, label="BIS_actual", c="r")
        plt.plot(y_pred, label="BIS_pred", c="b")
        plt.xlabel("Time")
        plt.ylabel("Time series values")
        plt.title("All labels with BIS index")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        print("TOTAL")
        plt.figure(figsize=(16,6))
        plt.plot(y.values, label="BIS_actual", c="r")
        plt.plot(y_total, label="BIS_pred", c="b")
        plt.xlabel("Time")
        plt.ylabel("Time series values")
        plt.title("All labels with BIS index")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        
    return mse, mse_Val


# In[141]:


mse, _ = calcmse(bis, pnim, co2, spo2, fc, 100, True,10)


# In[142]:


# #fine tune 

# i = 1
# mse = []
# m = 1e10
# ind = 10000
# while(i<51):
#     _,k = calcmse(bis, pnim,co2,spo2,fc,i,False,100)
#     if(k < m):
#         m = k
#         ind = i
#     mse.append(k)
#     print(i)
#     i = i+1

# print("min arg width = ", ind)

# plt.figure(figsize=(16,6))

# plt.plot(mse, label="MSE", c="r")

# plt.xlabel("width")
# plt.ylabel("MSE")
# plt.title("All labels with BIS index")

# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()


# In[143]:


# calcmse(bis, pnim,co2,spo2,fc,ind,True,100)


# In[144]:


import pandas as pd
import glob
import os
import random

# Corrected the path and pattern to match .parquet files
path = r'/Users/abhinavrajput/Desktop/PROJECTS/NoEEG_Borealis/doa-zero-eeg-sample_filtered'
all_files = glob.glob(os.path.join(path, "*.parquet"))

li = []

for filename in all_files:
    # Removed the index_col and header parameters
    df = pd.read_parquet(filename)
    li.append(df)

random.shuffle(li)

#frame = pd.concat(li, axis=0, ignore_index=True)


# In[145]:


######## K FOLD TRAINING #######

def kfold(lst, k):
    n = len(lst)
    slice_size = n // k  
    remainder = n % k  
    slices = []
    start = 0
    for i in range(k):
        end = start + slice_size + (1 if remainder > 0 else 0)
        remainder -= 1
        slices.append(lst[start:end])
        start = end
    return slices


# In[146]:


data = kfold(li, 5)


# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # Add this import
from sklearn.metrics import r2_score
import xgboost as xgb

def calcmse2(bis, pnim, co2, spo2, fc, bist, pnimt, co2t, spo2t, fct, width=42, want_to_print = False, n_estim=4):
    df = pd.DataFrame({
        'A': bis.flatten(), 
        'B': pnim.flatten(), 
        'C': co2.flatten(), 
        'D': spo2.flatten(), 
        'E': fc.flatten()
    })

    dft = pd.DataFrame({
        'A': bist.flatten(), 
        'B': pnimt.flatten(), 
        'C': co2t.flatten(), 
        'D': spo2t.flatten(), 
        'E': fct.flatten()
    })

    window_size = width

    for lag in range(1, window_size + 1):
        df[f'B_lag{lag}'] = df['B'].shift(lag)
        df[f'C_lag{lag}'] = df['C'].shift(lag)
        df[f'D_lag{lag}'] = df['D'].shift(lag)
        df[f'E_lag{lag}'] = df['E'].shift(lag)


    for lag in range(1, window_size + 1):
        dft[f'B_lag{lag}'] = dft['B'].shift(lag)
        dft[f'C_lag{lag}'] = dft['C'].shift(lag)
        dft[f'D_lag{lag}'] = dft['D'].shift(lag)
        dft[f'E_lag{lag}'] = dft['E'].shift(lag)

    df = df.dropna()
    dft = dft.dropna()

    features = [f'{var}_lag{lag}' for var in ['B', 'C', 'D', 'E',] for lag in range(1, window_size + 1)]
    X = df[features]
    y = df['A']

    Xt = dft[features]
    yt = dft['A']

    n = 0.9
    # Calculate the split index
    split_index = int(n * len(X))

    # Split the data into training and testing sets
    X_train, X_ = X[:split_index], X[split_index:]
    y_train, y_ = y[:split_index], y[split_index:]


    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estim)

    model.fit(X_train, y_train)

    y_pred = model.predict(Xt)
    y_total = model.predict(X)

    y_pred2 = model.predict(X_)

    mse = mean_squared_error(yt, y_pred)

    print("R2 score : ", r2_score(yt, y_pred))

    if(want_to_print):
        print("Temp")
        plt.figure(figsize=(16,6))
        #VISUALIZING A RANDOM PORTION OF TEST SURGERY
        plt.plot(y_.values[:2000], label="BIS_actual", c="r")
        plt.plot(y_pred2[:2000], label="BIS_pred", c="b")
        plt.xlabel("Time")
        plt.ylabel("Time series values")
        plt.title("All labels with BIS index")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        print("TEST")
        plt.figure(figsize=(16,6))
        #VISUALIZING A RANDOM PORTION OF TEST SURGERY
        plt.plot(yt.values[30000:32000], label="BIS_actual", c="r")
        plt.plot(y_pred[30000:32000], label="BIS_pred", c="b")
        plt.xlabel("Time")
        plt.ylabel("Time series values")
        plt.title("All labels with BIS index")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        print("TOTAL")
        plt.figure(figsize=(16,6))
        plt.plot(y.values, label="BIS_actual", c="r")
        plt.plot(y_total, label="BIS_pred", c="b")
        plt.xlabel("Time")
        plt.ylabel("Time series values")
        plt.title("All labels with BIS index")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        
    return mse


# In[148]:


mse = 0
k = 5

i = 0
while i < k:
    testli = data[i]
    j = 0
    trainli = []
    while j < k:
        if (j == 1):
            pass
        else:
            trainli = trainli + data[j]
        j = j+1
    
    testdf = pd.concat(testli, axis=0, ignore_index=True)
    traindf = pd.concat(trainli, axis=0, ignore_index=True)

    bis_t,pnim_t,fc_t,spo2_t,co2_t = generate_arrays(testdf)
    bis,pnim,fc,spo2,co2 = generate_arrays(traindf)

    mse_ = calcmse2(bis, pnim, co2, spo2, fc, bis_t, pnim_t, co2_t, spo2_t, fc_t, width=50, want_to_print = True, n_estim=100)
    mse = mse + mse_
    print(mse_)

    i = i+1

print("AVG MSE ", mse/k)



# In[149]:


print("AVG MSE ", mse/k)


#Blockers:

# Extremely hard to visulaise full surgereis, tough to focus on parts where to see the performance on testing 
# MSE in terms of BIS value isnt really a great predictor of performance 
# prediction value should be more sensitive to small fluctualtions in  most of the plane region, its observed to saturate to a constant value in those region. 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




