# Columbia-edX-a3
K-means classifier unsupervised learning and the EM GMM classifier unsupervised learning.

The Expectation Maximization Gaussian Mixture Model will find the k-clusters in the data. 
There are known problems of this algorithm with one of its clusters falling onto a single point 
and causing its covariance matrix to become singular. To combat this, it is suggested that 
if a cluster's mean falls directly onto a single point that you randomly rechoose its 
mean and an arbitrarily high covariance matrix. 

The dataset is the iris dataset at UCI Machine Learning repository.

In unsupervised learning we do not have labeled data. We try and group the data and find the hidden structure.

![alt text](https://github.com/frogger21/columbia-edX-a3/blob/master/edx4.JPG)

Source: Pattern Recognition and Machine Learning 2006
