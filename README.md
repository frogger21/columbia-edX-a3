# columbia-edX-a3
k-means classifier unsupervised learning and the EM GMM classifier unsupervised learning.

The Expectation Maximization Gaussian Mixture Model will find the k-clusters of the data. 
THere are known problems of this algorithm with one of its clusters falling onto a single point 
and causing its covariance matrix to become singular. To combat that it is suggested that 
if a cluster's mean falls directly onto a single point that you randomly rechoose its 
mean and an arbitrarily high covariance matrix. 

The dataset is the iris dataset at UCI Machine Learning repository.
