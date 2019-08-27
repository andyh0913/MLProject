import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import numpy as np
from gensim.models import Word2Vec
import keras
import jieba
import pickle
import os

def Instancenorm(x, gamma, beta):
	# x_shape:[B, H, W, C]
	results = 0.
	eps = 1e-5

#     x_mean = np.mean(x, axis=(1, 2), keepdims=True)
#     x_var = np.var(x, axis=(1, 2), keepdims=True)
	x_mean, x_var = tf.nn.moments(x, axes=(1,2), keep_dims=True)
	x_normalized = (x - x_mean) / tf.sqrt(x_var + eps)
	results = gamma * x_normalized + beta
	return results

class AugCycleGan():
	def __init__(self, mode, learning_rate, embedding_size, hidden_size, rnn_size, latent_size):
		self.mode = mode
		self.max_timestep = 25
		self.learning_rate = learning_rate
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.rnn_size = rnn_size
		self.latent_size = latent_size
		self.load_data()
		self.build_model()
		
	# define network models
	def get_generator(self, input_x, latent_z, sentence_lengths, reuse, name):
		with tf.variable_scope(name, reuse=reuse):
			conv1 = tf.layers.conv2d(input_x, 100, kernel_size=(3,self.embedding_size)) # (-1, h-3+1, 1, 100)
			conv2 = tf.layers.conv2d(input_x, 100, kernel_size=(4,self.embedding_size))
			conv3 = tf.layers.conv2d(input_x, 100, kernel_size=(5,self.embedding_size))
			gamma = tf.layers.dense(latent_z, 1)
			beta = tf.layers.dense(latent_z ,1)
			conv1 = Instancenorm(conv1, gamma, beta)
			conv2 = Instancenorm(conv2, gamma, beta)
			conv3 = Instancenorm(conv3, gamma, beta)
			conv1 = tf.tanh(conv1)
			conv2 = tf.tanh(conv2)
			conv3 = tf.tanh(conv3)
			max1 = tf.reduce_max(conv1, axis=(1,2))
			max2 = tf.reduce_max(conv2, axis=(1,2))
			max3 = tf.reduce_max(conv3, axis=(1,2))

			feature_vec = tf.concat([max1, max2, max3], 1)
			print (feature_vec)
# 			conv_output = tf.layers.dense(feature_vec, 1) # l2 norm applied in text-cnn paper
# 			output = tf.layers.dropout(output, 0.2) # also applied in text-cnn paper

			# define LSTM decoder
			def initial_fn():
				initial_elements_finished = (0 >= sentence_lengths) # sentence_lengths not defined yet
				bos_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, 1), 0) # id of BOS = 1
				initial_input = tf.tile(bos_embedding, [self.batch_size, 1])
				return initial_elements_finished, initial_input
			
			def sample_fn(time, outputs, state):
# 				custom helper 可以接 output layer嗎
# 				他的範例看起來是已經接了
				prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
				
				return prediction_id

			def next_inputs_fn(time, outputs, state, sample_id):
				output_prob = tf.softmax(outputs, axis=-1)
				next_embedding = output_prob.dot(embedding)
# 				elements_finished = (time >= sentence_lengths)
				elements_finished = tf.tile(tf.constant([0]), [self.batch_size])
# 				all_finished = tf.reduce_all(elements_finished) # making length different
# 				all_finished = False
				next_inputs = next_embedding
				next_state = state
				return elements_finished, next_inputs, next_state
			
			custom_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)
			output_layer = tf.layers.Dense(self.vocab_size)
# 			max_target_sequence_length = tf.reduce_max(sentence_lengths)
# 			mask = tf.sequence_mask(sentence_lengths, max_target_sequence_length, dtype=tf.float32)
			
# 			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
# 				num_units = self.rnn_size,
# 				memory = 
# 			)
			
			if self.mode == 'train':
				cell = tf.contrib.rnn.LSTMCell(self.rnn_size, state_is_tuple=True)
				decoder_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob_placeholder)
				training_decoder = tf.contrib.seq2seq.BasicDecoder(
					cell=decoder_cell,
					helper = custom_helper,
# 					initial_state = tf.concat((feature_vec, tf.random_uniform((1, self.embedding_size))), 1), # BOS_embedding
                    initial_state = LSTMStateTuple(feature_vec, tf.random_uniform((1, self.embedding_size))),
					output_layer = output_layer
				)
				decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
					decoder = training_decoder,
					impute_finished=True,
					maximum_iterations=self.max_timestep
				)
				
# 				# Calculate loss with sequence_loss
				decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
				self.decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1)
# 				self.loss = tf.contrib.seq2seq.sequence_loss(
# 					logits = decoder_logits_train,
# 					targets=self.decoder_targets,
# 					weights=mask
# 				)
# 				optimizer = tf.train.AdamOptimizer()
				return self.decoder_predict_train

			
	def get_discriminator(self, input_x, reuse, name):
		with tf.variable_scope(name, reuse=reuse):
			conv1 = tf.layers.conv2d(input_x, 100, kernel_size=(3,self.embedding_size)) # (-1, h-3+1, 1, 100)
			conv2 = tf.layers.conv2d(input_x, 100, kernel_size=(4,self.embedding_size))
			conv3 = tf.layers.conv2d(input_x, 100, kernel_size=(5,self.embedding_size))
			gamma = tf.layers.dense(latent_z, 1)
			beta = tf.layers.dense(latent_z ,1)
			conv1 = Instancenorm(conv1, gamma, beta)
			conv2 = Instancenorm(conv2, gamma, beta)
			conv3 = Instancenorm(conv3, gamma, beta)
			conv1 = tf.tanh(conv1)
			conv2 = tf.tanh(conv2)
			conv3 = tf.tanh(conv3)
			max1 = tf.reduce_max(conv1, axis=(1,2))
			max2 = tf.reduce_max(conv2, axis=(1,2))
			max3 = tf.reduce_max(conv3, axis=(1,2))

			feature_vec = tf.concat([max1, max2, max3], 1)
			conv_output = tf.layers.dense(feature_vec, 1, activation='sigmoid')
			
			return conv_output
			
			
	def get_encoder(self, input_x, input_y, reuse, name):
		with tf.variable_scope(name, reuse=reuse):
			input_concat = tf.concat(input_x, input_y, axis=1)
			conv1 = tf.layers.conv2d(input_concat, 100, kernel_size=(3,self.embedding_size)) # (-1, h-3+1, 1, 100)
			conv2 = tf.layers.conv2d(input_concat, 100, kernel_size=(4,self.embedding_size))
			conv3 = tf.layers.conv2d(input_concat, 100, kernel_size=(5,self.embedding_size))
			conv1 = tf.tanh(conv1)
			conv2 = tf.tanh(conv2)
			conv3 = tf.tanh(conv3)
			max1 = tf.reduce_max(conv1, axis=(1,2))
			max2 = tf.reduce_max(conv2, axis=(1,2))
			max3 = tf.reduce_max(conv3, axis=(1,2))
			
			feature_vec = tf.concat([max1, max2, max3], 1)
			conv_output = tf.layers.dense(feature_vec, self.latent_size)
			
			return conv_output
		
	def get_z_discriminator(self, latent_z, reuse, name):
		with tf.variable_scope(name, reuse=reuse):
			dense = tf.layers.dense(latent_z, 256, activation='relu')
			dense = tf.layers.dense(dense, 128, activation='relu')
			dense = tf.layers.dense(dense, 64, activation='relu')
			output = tf.layers.dense(dense, 1, activation='sigmoid')
			
			return output
	
	# get parameter-list by the following codes
	# d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
	# g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
	
	def load_data(self):
		ptt_data_path = 'data/Gossiping-Chinese-Corpus/ptt.pkl'
		news_data_path = 'data/news.pkl'
		with open(ptt_data_path, 'rb') as f:
			ptt_data = pickle.load(f)	
		with open(news_data_path, 'rb') as f:
			news_data = pickle.load(f)
# 		print (ptt_data[0:10])
# 		print (news_data[0:10])
		total_data = ptt_data + news_data
		
		if not os.path.exists("w2v_model"):
			print ("Building w2v_model...")
			w2v_model = Word2Vec(size=self.embedding_size, min_count=3, workers=8)
			w2v_model.build_vocab(total_data)
			w2v_model.train(total_data, total_examples=w2v_model.corpus_count, epochs=30)
			w2v_model.save("w2v_model")
		else:
			w2v_model = Word2Vec.load("w2v_model")
		self.w2v_model = w2v_model
		self.vocab_size = len(w2v_model.wv.vocab)

		
		# usage
		# print (w2v_model.wv.vocab['作業'].index)
		# print(w2v_model.wv.vectors) # embedding_matrix
		# print(w2v_model.wv.index2word) # ['我', '作業', '很', '多']
		
		embedding_matrix = w2v_model.wv.vectors
		special_embedding = np.zeros((4,self.embedding_size))
		# <PAD>: 0
		# <BOS>: 1
		# <EOS>: 2
		# <UNK>: 3
		special_embedding[1:4] = np.random.rand(3, self.embedding_size)
		
		ptt_data_idx = []
		news_data_idx = []
		
		if not os.path.exists('ptt_data_idx.npy'):
			for sentence in ptt_data:
				new_sentence = np.zeros(self.rnn_size)
				for i,word in enumerate(sentence[:self.rnn_size]):
					new_sentence[i] = self.word2index(word)
				if len(sentence) < self.rnn_size:
					new_sentence[len(sentence)] = 2
				else:
					new_sentence[self.rnn_size-1] = 2
				ptt_data_idx.append(new_sentence)
			np.save('ptt_data_idx', np.array(ptt_data_idx))
		else:
			ptt_data_idx = np.load('ptt_data_idx.npy')
# 		print (ptt_data_idx[0:10])
		
		if not os.path.exists('news_data_idx.npy'):
			for sentence in news_data:
				new_sentence = np.zeros(self.rnn_size)
				for i,word in enumerate(sentence[:self.rnn_size]):
					new_sentence[i] = self.word2index(word)
				if len(sentence) < self.rnn_size:
					new_sentence[len(sentence)] = 2
				else:
					new_sentence[self.rnn_size-1] = 2
				news_data_idx.append(new_sentence)
			np.save('news_data_idx', np.array(news_data_idx))
		else:
			news_data_idx = np.load('news_data_idx.npy')
# 		print (news_data_idx[0:10])
			
				
	def word2index(self, word):
		if(word == '<PAD>'):
			return 0
		elif(word == '<BOS>'):
			return 1
		elif(word == '<EOS>'):
			return 2
		elif(word == '<UNK>'):
			return 3
		elif (self.w2v_model.wv.vocab.get(word)):
			return self.w2v_model.wv.vocab[word].index + 4
		else:
			return 3
				

	
	# define optimizers
	def build_model(self):
		print('Building model...')
		
		# define placeholders
		self.batch_size = tf.placeholder(tf.int32)
		self.inputs_A = tf.placeholder(tf.int32, shape=(None, self.rnn_size)) # sentences from domain A
		self.inputs_B = tf.placeholder(tf.int32, shape=(None, self.rnn_size)) # sentences from domain B
		self.inputs_z_A = tf.placeholder(tf.float32, shape=(None, self.latent_size)) # latent vector for A
		self.inputs_z_B = tf.placeholder(tf.float32, shape=(None, self.latent_size)) # latent vector for B
		self.inputs_A_lengths = tf.placeholder(tf.int32, shape=[None]) # sentence lengths from domain A
		self.inputs_B_lengths = tf.placeholder(tf.int32, shape=[None]) # sentence lengths from domain B
		
		self.keep_prob_placeholder = tf.placeholder(dtype=tf.float32)
		
		# define embeddings (gensim word2vec)
		w2v_model = Word2Vec.load("w2v_model")
		w2v_embedding = np.zeros([len(w2v_model.wv.index2word), self.embedding_size])
		for i, word_idx in enumerate(w2v_model.wv.index2word):
			w2v_embedding[i, :] = w2v_model[word_idx]
		self.embeddings = tf.Variable(tf.convert_to_tensor(w2v_embedding, np.float32), trainable=False, name='embeddings')
		
		# define embedded inputs
		self.real_A = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, self.inputs_A), -1)
		self.real_B = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, self.inputs_B), -1)
		self.real_z_A = self.inputs_z_A
		self.real_z_B = self.inputs_z_B
		
		# inputs
# 		self.real_A = self.inputs_A
# 		self.real_B = self.inputs_B
# 		self.real_z_A = self.inputs_z_A
# 		self.real_z_B = self.inputs_z_B
		
		# A => B => A
		# networks
		self.fake_B = self.get_generator(self.real_A, self.real_z_B, self.inputs_A_lengths, reuse=False, name='generatorA_B')
		self.fake_z_A = self.get_encoder(self.real_A, self.fake_B, reuse=False, name='encoder_A')
		self.fake_A_ = self.get_generator(self.fake_B, self.fake_z_A, self.inputs_B_lengths, reuse=False, name='generatorB_A')
		self.fake_z_B_ = self.get_encoder(self.fake_B, self.real_A, reuse=False, name='encoder_B')
		self.D_real_B = self.get_discriminator(self.real_B, reuse=False, name='discriminator_B')
		self.D_fake_B = self.get_discriminator(self.fake_B, reuse=True, name='discriminator_B')
		self.D_real_z_A = self.get_z_discriminator(self.real_z_A, reuse=False, name='discriminator_z_A')
		self.D_fake_z_A = self.get_z_discriminator(self.fake_z_A, reuse=True, name='discriminator_z_A')
		
		# loss
		# D loss, maximized by G, minimized by D
		self.D_loss_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_B, labels=tf.ones_like(self.D_real_B)) + 
										 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_B, labels=tf.zeros_like(self.D_fake_B)))
		# D loss z, maximized by G, minimized by D
		self.D_loss_z_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_z_A, labels=tf.ones_like(self.D_real_z_A)) + 
										 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_z_A, labels=tf.zeros_like(self.D_fake_z_A)))
		
		# G loss, maximized by D, minimized by G
		self.G_loss_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_B, labels=tf.ones_like(self.D_fake_B)))
		# G loss z, maximized by D, minimized by G
		self.G_loss_z_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_z_A, labels=tf.ones_like(self.D_fake_z_A)))
		
		# Cycle consistency loss A
		max_inputs_A_length = tf.reduce_max(self.inputs_A_lengths)
		mask_A = tf.sequence_mask(self.inputs_A_lengths, max_inputs_A_length, dtype=tf.float32)
		self.CYC_loss_A = tf.reduce_mean(tf.sequence_loss( logits = self.fake_A_, targets = self.real_A, weights=mask_A))
		
		# Cycle consistency loss z_B
		self.CYC_loss_z_B = tf.reduce_mean(tf.reduce_sum(tf.abs(self.real_z_B - self.fake_z_B_)))
		
		# B => A => B
		# networks
		self.fake_A = self.get_generator(self.real_B, self.real_z_A, self.inputs_B_lengths, reuse=True, name='generatorB_A')
		self.fake_z_B = self.get_encoder(self.real_B, self.fake_A, reuse=True, name='encoder_B')
		self.fake_B_ = self.get_generator(self.fake_A, self.fake_z_B, self.inputs_A_lengths, reuse=True, name='generatorA_B')
		self.fake_z_A_ = self.get_encoder(self.fake_A, self.real_B, reuse=True, name='encoder_A')
		self.D_real_A = self.get_discriminator(self.real_A, reuse=False, name='discriminator_A')
		self.D_fake_A = self.get_discriminator(self.fake_A, reuse=True, name='discriminator_A')
		self.D_real_z_B = self.get_z_discriminator(self.real_z_B, reuse=False, name='discriminator_z_B')
		self.D_fake_z_B = self.get_z_discriminator(self.fake_z_B, reuse=True, name='discriminator_z_B')
		
		# loss
		# D loss, maximized by G, minimized by D
		self.D_loss_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_A, labels=tf.ones_like(self.D_real_A)) + 
										 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_A, labels=tf.zeros_like(self.D_fake_A)))
		# D loss z, maximized by G, minimized by D
		self.D_loss_z_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_z_B, labels=tf.ones_like(self.D_real_z_B)) + 
										   tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_z_B, labels=tf.zeros_like(self.D_fake_z_B)))
		
		# G loss, maximized by D, minimized by G
		self.G_loss_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_A, labels=tf.ones_like(self.D_fake_A)))
		# G loss z, maximized by D, minimized by G
		self.G_loss_z_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_z_B, labels=tf.ones_like(self.D_fake_z_B)))
		
		# Cycle consistency loss B
		max_inputs_B_length = tf.reduce_max(self.inputs_B_lengths)
		mask_B = tf.sequence_mask(self.inputs_B_lengths, max_inputs_B_length, dtype=tf.float32)
		self.CYC_loss_B = tf.reduce_mean(tf.sequence_loss( logits = self.fake_B_, targets = self.real_B, weights=mask_B))
		
		# Cycle consistency loss z_A
		self.CYC_loss_z_A = tf.reduce_mean(tf.reduce_sum(tf.abs(self.real_z_A - self.fake_z_A_)))
		
		

		# get var_lists
		D_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_A')
		D_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_B')
		D_z_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_z_A')
		D_z_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_z_B')
		
		G_B_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generatorB_A')
		G_A_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generatorA_B')
		E_A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder_A')
		E_B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder_B')
		
		self.D_loss = self.D_loss_A + self.D_loss_B + self.D_loss_z_A + self.D_loss_z_B
		self.G_loss = self.G_loss_A + self.G_loss_B + self.G_loss_z_A + self.G_loss_z_B + self.CYC_loss_A + self.CYC_loss_B + self.CYC_loss_z_A + self.CYC_loss_z_B
		
		self.D_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize( self.D_loss, var_list=D_A_vars + D_B_vars + D_z_A_vars + D_z_B_vars )
		self.G_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize( self.G_loss, var_list=G_B_A_vars + G_A_B_vars + E_A_vars + E_B_vars )
		
	def train(self, epochs, batch_size, g_iter, d_iter, model_dir):
		print ("Training...")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print (ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ("Reloading model parameters...")
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			for epoch in range(epochs):
				sample_latent_z_A = np.random.normal(size=(batch_size, self.latent_size))
		