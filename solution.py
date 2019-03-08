import numpy as np
import sys
from helper import *



def third_order(X):
	"""Third order polynomial transform on features X.
	 tranform (1,x1,x2) => (1,x1,x2,x1^2,x1x2,x2^2,x1^3,x1^2 x2,x1 x2^2,x2^3)

	Args:
		X: An array with shape [n_samples, 2].

	Returns:
		poly: An (numpy) array with shape [n_samples, 10].
	"""

	### YOUR CODE HERE

	#initialize empty array with shape (n_samples, 10)
	# list of list for transform points
	# it was from slides
	transformed_array = []
	for x in X :
		transformed_array.append([1, x[1],x[2], pow(x[1],2), x[1]*x[2], pow(x[2],2),pow(x[1],3), pow(x[1],2)* x[2], x[1] * pow(x[2], 2), pow(x[2],3)])

	return np.asanyarray(transformed_array)

	### END YOUR CODE


class LogisticRegression(object):
	
	def __init__(self, max_iter, learning_rate, third_order=False):
		self.max_iter = max_iter
		self.lr = learning_rate
		self.third_order = third_order

	#logistic regression
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))


	def _gradient(self, X, y):
		"""Compute the gradient with samples (X, y) and weights self.W.

		Args:
			X: An array with shape [n_samples, n_features].
			   (n_features depends on whether third_order is applied.)
			y: An array with shape [n_samples,]. Only contains 1 or -1.

		Returns:
			gradient: An array with shape [n_features,].

		"""
		### YOUR CODE HERE

		val = np.zeros(X.shape[1])

		for xn, yn in zip(X, y) :
			a = yn * xn
			b = 1 + np.exp(yn * (np.dot(self.W, xn)))
			val += a / b
		return -val / len(val)
		### END YOUR CODE


	def fit(self, X, y):
		"""Train logistic regression model on data (X,y).
		(If third_order is true, do the 3rd order polynomial transform)


		Args:
			X: An array with shape [n_samples, 3].
			y: An array with shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE


		if self.third_order:
			X = third_order(X)

		self.W = np.zeros(X.shape[1])

		for i in range(self.max_iter):
			v = -self._gradient(X, y)
			self.W = self.W + (self.lr * v)


		### END YOUR CODE
		return self


	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
			   (n_features depends on whether third_order is applied.)
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W


	def predict(self, X):
		"""Predict class labels for samples in X.
		(If third_order is true, do the 3rd order polynomial transform)

		Args:
			X: An array of shape [n_samples, 3].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
		if self.third_order:
			X = third_order(X)

		return np.where(self.sigmoid(np.dot(X, self.W)) >= 0.5, 1, -1)


		### END YOUR CODE


	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: A float. Mean accuracy of self.predict(X) wrt. y.
		"""
		return np.mean(self.predict(X)==y)



def accuracy_logreg(max_iter, learning_rate, third_order, 
					X_train, y_train, X_test, y_test):

	# train perceptron
	model = LogisticRegression(max_iter, learning_rate, third_order)
	model.fit(X_train, y_train)
	train_acc = model.score(X_train, y_train)

	# test perceptron model
	test_acc = model.score(X_test, y_test)

	return train_acc, test_acc