from tkinter.tix import INTEGER
import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    # First set some variables for readability
    D = 784
    C = 10
    N = input.shape[0]
    inW = np.dot(input, W)
    sigmoidInW = sigmoid(inW)
                
    # Next fill out loss
    # As we're already calculating prediction, it is more efficient to use sparse multi-class cross entropy loss instead of normal cross entropy loss.
    loss = 0
    #(-1 * label * np.log(np.dot(input, W)))
    individualLoss = -1 * label * np.log(sigmoidInW)
    for i in individualLoss:
      for j in i:
        loss += j
    loss /= individualLoss.size
    
    # Next fill out gradient
    # Start by setting needed variables for cleanliness
    regularizationTerm = ((lamda/N) * W)
    inputT = np.transpose(input)
    # Combine into gradient
    gradient =  ((1/N) * np.dot(inputT, (sigmoidInW - label)) + regularizationTerm)

    # Next calculate prediction
    # Loop through input 
    prediction = np.zeros(N)
    for i in range(N):
      max = input[i, 0]
      for j in range(C):
        if(sigmoidInW[i, j] > max):
          prediction[i] = j
          max = sigmoidInW[i, j]

    ############################################################################

    return loss, gradient, prediction
