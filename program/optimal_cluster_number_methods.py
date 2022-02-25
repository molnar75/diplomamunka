from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time

# Method for calculating the Silhouette score and measuring run time
def silhouette_method(values, labels):
    start = time.time()
    s_score = silhouette_score(values, labels)
    end = time.time()
    
    s_time = end-start
    
    return s_time, s_score

# Method for calculating the Davies-Bouldin score and measuring run time
def davies_bouldin_method(values, labels):
    start = time.time()
    db_score = davies_bouldin_score(values, labels)
    end = time.time()

    db_time = end-start
    
    return db_time, db_score

# Method for calculating the Calinski-Harabasz score and measuring run time
def calinski_harabasz_method(values, labels):
    start = time.time()
    ch_score = calinski_harabasz_score(values, labels)
    end = time.time()
    
    ch_time = end-start
    
    return ch_time, ch_score