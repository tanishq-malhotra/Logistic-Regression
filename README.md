## Implementing Logistic Regression from Scratch using just numpy

Logistic Regression is a Classification algorithm which classifies either 0 or 1 | true or false.

A full implementation with graphical plots is given in the jupyter notebook.

A test file is also included showing the normal implementation.

# Hypothesis
 
 A function which takes inputs and returns outputs. Logistic regression uses a hypothesis which 
 ouputs a value betwenn 0 or 1.

 So the Hypothesis used here is:
![alt text][graph]

[graph]: https://cdn-images-1.medium.com/max/1100/1*HXCBO-Wx5XhuY_OwMl0Phw.png

![alt text][func]

[func]: https://cdn-images-1.medium.com/max/1100/1*p4hYc2VwJqoLWwl_mV0Vjw.png

# Loss Function

A loss function is a function which describes how bad our parameters is for our model or how
bad our model is working. 

We Cannot use the mean squared error function here with this hypothesis because it gives a non-convex output,
in which there are many local minina and gradient descent fail to run on it.
So we want a new loss function here.

## So the loss function used in Logistic Regression is:
![alt text][loss]

[loss]: https://cdn-images-1.medium.com/max/1375/1*FdxEs8Iv_43Q8calTCjnow.png

# Gradient Descent on Logistic Regression
Our goal is to minimize the loss function and the way we have to achive it is by increasing/decreasing the weights, i.e. fitting them.
So GD is derivative of the loss function with respect to each weight. 
It tells us how loss would change if we modified the parameters.

## Gradient Descent:
![alt text][gd]

[gd]: https://cdn-images-1.medium.com/max/1100/1*gobKgGbRWDAoVFAan_HjxQ.png