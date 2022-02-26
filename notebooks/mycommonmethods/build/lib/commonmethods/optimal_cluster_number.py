from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time

def silhouette_method(values, labels):
    """
    Calculating the Silhouette method for the given values with the given labels, and measuring time.
    :param values: in my case the pixel values from the k-means method
    :param labels: in my case the labels from the k-means method
    :return: the time of the calculation and the calculated Silhouette score
    """
    start = time.time()
    s_score = silhouette_score(values, labels)
    end = time.time()
    
    s_time = end-start
    
    return s_time, s_score

def davies_bouldin_method(values, labels):
    """
    Calculating the Davies-Bouldin method for the given values with the given labels, and measuring time.
    :param values: in my case the pixel values from the k-means method
    :param labels: in my case the labels from the k-means method
    :return: the time of the calculation and the calculated Davies-Bouldin score
    """
    start = time.time()
    db_score = davies_bouldin_score(values, labels)
    end = time.time()

    db_time = end-start
    
    return db_time, db_score

def calinski_harabasz_method(values, labels):
    """
    Calculating the Calinski-Harabasz method for the given values with the given labels, and measuring time.
    :param values: in my case the pixel values from the k-means method
    :param labels: in my case the labels from the k-means method
    :return: the time of the calculation and the calculated Calinski-Harabasz score
    """
    start = time.time()
    ch_score = calinski_harabasz_score(values, labels)
    end = time.time()
    
    ch_time = end-start
    
    return ch_time, ch_score