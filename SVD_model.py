# -*- coding: utf-8 -*-
"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.base.wrapper as wrap
from statsmodels.tsa.base.tsa_model import (
    TimeSeriesResultsWrapper,
)
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.vector_ar import util
from statsmodels.base.data import handle_data


def chatgpt_forecast(y, coefs, lags, steps):
    """
    Produce linear minimum MSE forecast
    Parameters
    ----------
    y : ndarray (k_ar x neqs)
        The past values of the time series.
        k_ar is the number of lags used in the model
        neqs is the number of variables in the system.
    coefs : ndarray (k_ar x neqs x neqs)
        The coefficients of the model.
        k_ar is the number of lags used in the model
        neqs is the number of variables in the system.
    steps : int
        The number of steps to forecast into the future.
    Returns
    -------
    forecasts : ndarray (steps x neqs)
        The forecasted values for the next steps.
    Notes
    -----
    Lütkepohl p. 37
    """
    # Get the number of variables in the system
    neqs = y.shape[1]

    # Initialize the forecast array with zeros
    forecasts = np.zeros((steps, neqs))

    y_forecasts = np.zeros((steps + lags, neqs))
    y_forecasts[:lags, :] = y[-lags:, :]
    #y_forecasts = np.zeros((steps * 2, neqs))
    y#_forecasts[:steps, :] = y
    coefs_flat = coefs.ravel(order="F")
    # neqs_sq = neqs * neqs
    
    # Loop through the forecast steps
    lags = lags
    for h in range(steps):
        f = forecasts[h]
        for i in range(1, lags + 1):
            prior_y = y_forecasts[lags + h - i]
            for j in range(neqs):
                
                idx = j * lags * neqs + (i - 1) * neqs
                
                f[j] += np.dot(coefs_flat[idx : idx + neqs], prior_y)

        # Update y with the forecast for the next step
        y_forecasts[lags + h] = f
        forecasts[h] = f

    return forecasts


def forecast(y, coefs, steps):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
    coefs : ndarray (k_ar x neqs x neqs)
    steps : int

    Returns
    -------
    forecasts : ndarray (steps x neqs)

    Notes
    -----
    Lütkepohl p. 37
    """
    p = len(coefs)
    k = len(coefs[0])
    # print(f"{coefs.shape = }")
    # print(f"{p = }")
    # print(f"{k = }")
    # initial value
    forcs = np.zeros((steps, k))
    # print(f"{forcs.shape = }")
    # print(f"{coefs.shape = }")

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]
            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)
        # print(f.shape)
        forcs[h - 1] = f

    return forcs


class ShavedVAR:
    r"""
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable. The independent variable.

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """

    def __init__(self, endog):
        self.endog = endog
        self.neqs = self.endog.shape[1]

    def fit(
        self,
        maxlags: int,
        alpha: int,
        method = 'SVD'
    ):
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : {int, None}, default None
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        alpha : float
            L2 regularization constant
        method : string
            Which method is used to find parameters, SVD is default
            
        Returns
        -------
        VARResults
            Estimation results

        Notes
        -----
        See Lütkepohl pp. 146-153 for implementation details.
        """
        lags = maxlags

        return self._estimate_var(lags, alpha, method=method)

    def _estimate_var(self, lags, alpha, method):
        """
        lags : int
            Lags of the endogenous variable.
        alpha : float
            L2 regularization constant
        method : string
            Which method is used to find parameters, SVD is default
        """
        endog = self.endog
        nobs = len(endog)
    
        # Ravel C order, need to put in descending order
        # z = np.array([endog[t - lags : t][::-1].ravel() for t in range(lags, nobs)])
        lb_minus_lags = nobs - lags
        #Was originally (3 * 200 * lags)
        z = np.zeros((lb_minus_lags, endog.shape[1] * lags))
        #print(z.shape)
        for t in range(lags, nobs):
            t_minus_lags = t - lags
            middle_cols = endog[t_minus_lags:t, :][::-1, :].ravel()
            z[t_minus_lags, :] = middle_cols

        y_sample = endog[lags:]
        if method=="SVD":
            
            U, s, Vt = np.linalg.svd(z, full_matrices=False)
            # Create Sigma matrix from 1D array
            sigma = np.diag(s)
            # Create a Sigma squared matrix
            sigma_sq = np.dot(sigma, sigma)
            # Apply penalty to the singular values
            sigma_a = sigma_sq + np.identity(sigma.shape[0]) * alpha
            # Calculate inverse
            sigma_a_inv = np.diag(1/np.diag(sigma_a))
            UTy = np.dot(U.T, y_sample)
            Sigma_UTy = np.dot(sigma, UTy)
            sigma_penilized = np.dot(sigma_a_inv, Sigma_UTy)
           
            # Calculate coefficients
            params = np.dot(Vt.T, sigma_penilized)
            #params = np.dot(np.linalg.pinv(z), y_sample)
        elif method=="Inverse": 
            params = np.dot(np.linalg.inv(np.dot(z.T, z)+alpha*np.identity(lags*endog.shape[1])), np.dot(z.T, y_sample))
        else:
            print("ERROR: invalid method chosen. Please choose 'SVD' or 'Inverse' as the method")
        #params = np.linalg.lstsq(z, y_sample, rcond=None)[0]
        
        # neqs = self.endog.shape[1]
        # Initialize VARProcess parent class
        # construct coefficient matrices
        # Each matrix needs to be transposed
        # reshaped = params.reshape((lags, neqs, neqs))

        # Need to transpose each coefficient matrix
        # coefs = reshaped.swapaxes(1, 2)

        return VARProcess(params, lags)


class VARProcess:
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        coefficients for lags of endog, part or params reshaped
    """

    def __init__(self, coefs, max_lags):
        # self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.lags = max_lags

    def forecast(self, y, steps):
        """Produce linear minimum MSE forecasts for desired number of steps
        ahead, using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
        steps : int

        Returns
        -------
        forecasts : ndarray (steps x neqs)

        Notes
        -----
        Lütkepohl pp 37-38
        """
        return chatgpt_forecast(y, self.coefs, self.lags, steps)
