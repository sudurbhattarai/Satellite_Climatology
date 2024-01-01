def xr_trend(xarr, deltaT = 'day'):    
    """function that calculates the temporal trend in xarray DataArrays
    xarr is an xarray DataArray
    deltaT is the time unit for which the trend is calculated (can be 'day', 
    'year', or 'decade')
    returns an xarray Dataset containing the 'slope', 'pval' and 'intercept' 
    """
    from scipy import stats
    import numpy as np
    # getting shapes
    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]
    # creating x and y variables for linear regression
    x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
    y = xarr.to_masked_array().reshape(n, -1)
    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean
    ya = y - ym  # anomaly
    xa = x - xm  # anomaly
    # variance and covariances
    xss = (xa ** 2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya ** 2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    if deltaT == 'day':
        slope = slope
    elif deltaT == 'year':
        slope = slope * 365.25
    elif deltaT == 'decade':
        slope = slope * 3652.5 
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss)**0.5
    t = r * (df / ((1 - r) * (1 + r)))**0.5
    p = stats.distributions.t.sf(abs(t), df)
    # preparing outputs
    out = xarr[:2].mean('time')
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name = '_slope' # was +=
    xarr_slope.attrs['units'] = 'units / ' + deltaT
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name += '_Pvalue'
    xarr_p.attrs['info'] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # do the same for the intercept
    xarr_intercept = out.copy()
    xarr_intercept.name += '_intercept'
    xarr_intercept.attrs['units'] = 'units / ' + deltaT
    xarr_intercept.values = intercept.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name='slope')
    xarr_out['pval'] = xarr_p
    xarr_out['intercept'] = xarr_intercept
    return xarr_out

