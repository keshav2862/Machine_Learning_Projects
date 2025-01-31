import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W




def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return (1.0 / (1.0 + np.exp(-z)))



def sigmoid_derivative(z):
    sigm = 1.0 / (1.0 + np.exp(-z))
    return sigm * (1.0 - sigm)
    



def count_feature_indices(boolean_value):
    
    count = 0
    feature_indices = []
    
    for i in range(len(boolean_value)):
        if boolean_value[i]==False:
            count += 1
            feature_indices.append(i)
            print(i,end =" ")
    print(" ")
    print("Total number of selected features : ", count)




def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    X_data = []
    y_data = []
    for i in range(10):
        train_mat = mat['train'+ str(i)]
        labels = np.full((train_mat.shape[0],1),i)
        labeled_train_mat = np.concatenate((train_mat,labels),axis=1)
        X_data.append(labeled_train_mat)
        test_mat = mat['test'+ str(i)]
        labels_test = np.full((test_mat.shape[0],1),i)
        labeled_test_mat = np.concatenate((test_mat,labels_test),axis=1)
        y_data.append(labeled_test_mat)

    X_data_combined = np.concatenate((X_data[0],X_data[1],X_data[2],X_data[3],X_data[4],X_data[5],X_data[6],X_data[7],X_data[8],X_data[9]), axis=0)
    
    np.random.shuffle(X_data_combined)
    
    labeled_X = X_data_combined[0:50000,:]
    X_data_shuffled = labeled_X[:,0:784]
    X_data_label   = labeled_X[:,784]

    X_data_shuffled = X_data_shuffled / 255.0

    labeled_val = X_data_combined[50000:60000,:]
    val_data    = labeled_val[:,0:784] 
    val_label   = labeled_val[:,784]

    val_data = val_data / 255.0  

    y_data_combined = np.concatenate((y_data[0],y_data[1],y_data[2],y_data[3],y_data[4],y_data[5],y_data[6],y_data[7],y_data[8],y_data[9]), axis=0)

    np.random.shuffle(y_data_combined)
    
    y_data_shuffled    = y_data_combined[:,0:784]
    y_label_shuffled   = y_data_combined[:,784]

    y_data_shuffled = y_data_shuffled / 255.0
    
    combined  = np.concatenate((X_data_shuffled, val_data),axis=0)
    reference = combined[0,:]
    boolean_value_columns = np.all(combined == reference, axis = 0)
    
    count_feature_indices(boolean_value_columns)
    
    final = combined[:,~boolean_value_columns]
    
    tr_R = X_data_shuffled.shape[0]

    
    X_data_shuffled      = final[0:tr_R,:]
    val_data = final[tr_R:,:]
    y_data_shuffled = y_data_shuffled[:,~boolean_value_columns]
    
    print("preprocess done")
    return X_data_shuffled, X_data_label, val_data, val_label, y_data_shuffled, y_label_shuffled

def feedForward( X, w1, w2):
    n = X.shape[0]
    X_bias = np.concatenate((np.full((n,1), 1), X),axis=1)
    
    hidden_input = np.dot(X_bias, np.transpose(w1))
    
    hidden_output = sigmoid(hidden_input)
    
    m = hidden_output.shape[0]

    out_input_bias = np.concatenate((np.full((m,1), 1), hidden_output), axis=1)
    
    out_input = np.dot(out_input_bias, np.transpose(w2))
    out_output = sigmoid(out_input)
    return out_output, out_input_bias, X_bias, n
    


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0

    ol, hidden_bias, training_data_bias, n = feedForward(training_data, w1, w2)

    yl = np.full((n, n_class), 0)

    for i in range(n):
        trueLabel = training_label[i]
        yl[i][trueLabel] = 1
    
    negative_log = np.sum( np.multiply(yl,np.log(ol)) + np.multiply(1.0-yl,np.log(1.0-ol)) )/((-1)*n)
    
    diff = ol- yl
    grad2 = np.dot(diff.T, hidden_bias)
   
    temp = np.dot(diff,w2) * ( hidden_bias * (1.0-hidden_bias))
    
    grad1 = np.dot( np.transpose(temp), training_data_bias)
    grad1 = grad1[1:, :]
    
    regularization =  lambdaval * (np.sum(w1**2) + np.sum(w2**2)) / (2*n)
    obj_val = negative_log + regularization
    
    grad_w1 = (grad1 + lambdaval * w1)/n
    grad_w2 = (grad2 + lambdaval * w2)/n

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)




def nnPredict(w1, w2, training_data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    ol, ob, ib, n = feedForward(training_data, w1, w2)

    labels = np.argmax(ol, axis=1)
    return labels



"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]
  

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 30

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 200}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
