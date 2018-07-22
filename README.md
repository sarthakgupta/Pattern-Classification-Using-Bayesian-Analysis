# Pattern-Classification-Using-Bayesian-Analysis
This project as part of Course Basics of Signal Processing(E9 207) in IISc, Bengaluru. Here, SVD is used for pre-processing of data and then bayesian analysis is done for MNIST and CIFAR-10 dataset. With MNIST around 94% of test accuracy is achieved. With CIFAR-10 38% of test accuracy is achieved. Number of features to be selected after SVD is done using cross validation (though due to slower system instead of cross validation I am testing on test data only).
Real time Demo for Detection of Handwritten Digits: 
1. Run Bayesian_SVD_MNIST.py -> it creates Bayesian Model for 10 different classes (0 to 10) based on MNIST data.
2. Open Image Test_digit.png in paint -> draw the digit(0 to 9) using white calligraphy brush 1 with minimum thickness.
3. Run Image_converter.m -> Convert the image in matrix format as required by Bayesian Model. (This can be done in python also but due to some problem in installing package couldn't able to do it.)
4. Run Test_image.py -> predict the handwritten digit drawn.

Video Link: https://www.youtube.com/watch?v=ML2PR5Qtq8M&t=159s
