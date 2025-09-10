import numpy as np

def normalize_interval(data, method='dispersion'):
    """
    Normalizes interval-valued data based on the chosen method.
    
    Parameters
    ----------
    data : ndarray
        Interval-valued data.

    method : str, default='dispersion'
        Normalization method to apply. Supported methods include:
        - 'dispersion': Normalizes data to have zero mean and unit dispersion per feature.

    Returns
    -------
    normalized_data : ndarray
        Normalized interval-valued data with the same shape as the input.

    Raises
    ------
    ValueError
        If the input shape is not adequate or normalization causes a division by zero.
    """
    
    data = np.asarray(data)
    
    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError(f"Input data must be interval-valued")
    
    if method == 'dispersion':
        
        means = np.mean(data, axis=0)
        dispersions = np.sum((data-means)**2,axis=(0,2))

        if np.any(dispersions == 0):
            raise ValueError("Zero division")

        normalized_data = (data - means) / np.sqrt(dispersions)[np.newaxis,:,np.newaxis]
    
    return normalized_data

def normalize_radial(data, method='dispersion'): # currently equivalent to normalize_radial
    """
    Normalizes radial-valued data based on the chosen method.
    
    Parameters
    ----------
    data : ndarray
        Radial-valued data.

    method : str, default='dispersion'
        Normalization method to apply. Supported methods include:
        - 'dispersion': Normalizes data to have zero mean and unit dispersion per feature.

    Returns
    -------
    normalized_data : ndarray
        Normalized interval-valued data with the same shape as the input.

    Raises
    ------
    ValueError
        If the input shape is not adequate or normalization causes a division by zero.
    """

    data = np.asarray(data)
    
    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError(f"Input data must be radial-valued")

    means = np.mean(data, axis=0)
    dispersions = np.sum((data-means)**2, axis=0)
    
    if np.any(dispersions == 0):
            raise ValueError("Zero division")

    normalized_data = (data - means) / np.sqrt(dispersions)[np.newaxis,:,np.newaxis]

    return normalized_data

def interval_to_radial(data):
    """
    Converts interval data into center-radius form.

    Parameters
    ----------
    data : np.ndarray
        Input array in interval format

    Returns
    -------
    np.ndarray
        Output array in center-radius format
        
    """
    
    data = np.asarray(data)
    
    lower = data[:, :, 0]
    upper = data[:, :, 1]

    center = (lower + upper) / 2
    radius = (upper - lower) / 2

    return np.stack((center, radius), axis=2)
    