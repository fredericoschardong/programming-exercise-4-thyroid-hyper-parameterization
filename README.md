# Programming Exercise 4: feed forward single/multiple-hidden layer classifier for thyroid
Python (sklearn-based) implementation that explores how different parameters impact a feed-forward neural network with single/multiple hidden layer. 

A brief analysis of the results is [provided in Portuguese](https://github.com/fredericoschardong/programming-exercise-4-thyroid-hyper-parameterization/blob/master/report%20in%20portuguese.pdf). It was submited as an assignment of a graduate course named [Connectionist Artificial Intelligence](https://moodle.ufsc.br/mod/assign/view.php?id=2122514) at UFSC, Brazil.

In short, two normalization methods are evaluated (minmax and [Yeo-Johnson](https://doi.org/10.1093/biomet/87.4.954)) in a thyroid dataset [from UCI](http://archive.ics.uci.edu/ml/index.php) [ported to matlab](https://www.tamps.cinvestav.mx/~wgomez/downloads.html) with multiple training algorithms, hidden layers, learning rate (alpha), epochs and activation functions.

Normalization:
![](https://github.com/fredericoschardong/programming-exercise-4-thyroid-hyper-parameterization/blob/master/results/Histogram%2C%20before%2C%20normalization.png "")

