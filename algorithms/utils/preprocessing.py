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