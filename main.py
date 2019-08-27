import tensorflow as tf
import numpy as np
from augcyclegan import AugCycleGan

if __name__ == "__main__":
	model = AugCycleGan(mode = 'train', learning_rate = 0.001, embedding_size = 256, hidden_size = 300, rnn_size = 40, latent_size = 256)
