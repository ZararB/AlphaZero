from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout


class Network(object):

	def __init__(self, config):
		self.model = Sequential([

			]
			)

	def inference(self, image):
		model_input = preprocess_image(image)

		model_output = self.model.predict(model_input)
		value = model_output[0, 0]
		policy = model_output[0, 1]

    	return (value, policy)  # Value, Policy

  	def get_weights(self):
	    # Returns the weights of this network.
	    return []

	def preprocess_image(self, image):

		return model_input