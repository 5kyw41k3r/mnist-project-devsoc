# mnist-project-devsoc
repo for mnist project that i made


make sure train.csv and test.csv are in the same directory as the python files...
couldn't upload them cause github said 25MB file limit...

right now trainer runs 100 epochs and saves the model to a file called "trained_model_100epochs.npz" which is just multiple arrays of optimized weights and biases found by gradient descent.


the tester code on the other hand outputs mismatched elements from the labels which the ai failed to predict... 

it wasnt possible to do this with test.csv cause i couldnt test the accuracy(labels arent preasent) but if u want u can change it so it predicts numbers from test.csv as well
