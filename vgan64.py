import tensorflow as tf
import numpy as np
import h5py
import os
from layers import *

######################### settings ##############################
z_size = 100
batch_size = 64
IS_TRAIN = True
starter_learning_rate = 0.0002
beta1 = 0.5
train_iters = 100000
A,B,C = 64,64,32
channel = 3
img_size = 64*64
mode = "train"
restore = False
image_file_name = "./pretrain/images_npy/"
model_file_name = "./pretrain/model/"
dataset_filename = '/media/skye/b674e37c-7857-43ec-9527-dd932cb58935/beach0-64.hdf5'
restore_path = '/home/skye/Desktop/VGAN-Tensorflow/pretrain/model/'
log_path = "/home/skye/Desktop/VGAN-Tensorflow/pretrain/logs/"
###################################################################
if os.path.exists(image_file_name) == False:
	os.mkdir(image_file_name)
if os.path.exists(model_file_name) == False:
	os.mkdir(model_file_name)
if os.path.exists(log_path) == False:
	os.mkdir(log_path)
	
tf.reset_default_graph()

def next_batch(data_array):
    length=data_array.shape[0] #assuming the data array to be a np arry
    permutations=np.random.permutation(length)
    idxs=permutations[0:batch_size]
    batch=np.zeros([batch_size, C, img_size*channel], dtype=np.float32)
    for i in range(len(idxs)):
        batch[i,:]=data_array[idxs[i]].reshape(C,img_size*channel)
    return (batch / 255 - 0.5) / 0.5

def static_net(z):
		static_input = tf.reshape(z,[batch_size, 1, 1, z_size])

		static_dconv1 = deconv2d(static_input,[4,4,512,z_size],[batch_size, 4,4,512],[1,1,1,1],'VALID',name='st_deconv1')
		static_dconv1 = batch_norm2(static_dconv1,scope='st_bn_dconv1',num_of_filters=512)
		static_dconv1 = tf.nn.relu(static_dconv1)

		static_dconv2 = deconv2d(static_dconv1,[4,4,256,512],[batch_size, 8,8,256],[1,2,2,1],'SAME',name='st_deconv2')
		static_dconv2 = batch_norm2(static_dconv2,scope='st_bn_dconv2',num_of_filters=256)
		static_dconv2 = tf.nn.relu(static_dconv2)

		static_dconv3 = deconv2d(static_dconv2,[4,4,128,256],[batch_size, 16,16,128],[1,2,2,1],'SAME',name='st_deconv3')
		static_dconv3 = batch_norm2(static_dconv3,scope='st_bn_dconv3',num_of_filters=128)
		static_dconv3 = tf.nn.relu(static_dconv3)

		static_dconv4 = deconv2d(static_dconv3,[4,4,64,128],[batch_size, 32, 32,64],[1,2,2,1],'SAME',name='st_deconv4')
		static_dconv4 = batch_norm2(static_dconv4,scope='st_bn_dconv4',num_of_filters=64)
		static_dconv4 = tf.nn.relu(static_dconv4)

		static_dconv5 = deconv2d(static_dconv4,[4,4,3,64],[batch_size, 64,64, 3],[1,2,2,1],'SAME',name='st_deconv5')
		static_dconv5 = tf.nn.tanh(static_dconv5)

		return static_dconv5


def video_net_and_mask(z):
		video_input = tf.reshape(z,[batch_size, 1, 1, 1, z_size])

		video_dconv1 = deconv3d(video_input,[2,4,4,512,z_size],[batch_size, 2, 4,4,512],[1,1,1,1,1],'VALID',name='video_deconv1')
		video_dconv1 = batch_norm2(video_dconv1,scope='vd_bn_dconv1',num_of_filters=512)
		video_dconv1 = tf.nn.relu(video_dconv1)

		video_dconv2 = deconv3d(video_dconv1,[4,4,4,256,512],[batch_size, 4, 8, 8, 256],[1,2,2,2,1],'SAME',name='video_deconv2')
		video_dconv2 = batch_norm2(video_dconv2,scope='vd_bn_dconv2',num_of_filters=256)
		video_dconv2 = tf.nn.relu(video_dconv2)

		video_dconv3 = deconv3d(video_dconv2,[4,4,4,128,256],[batch_size, 8, 16, 16, 128],[1,2,2,2,1],'SAME',name='video_deconv3')
		video_dconv3 = batch_norm2(video_dconv3,scope='vd_bn_dconv3',num_of_filters=128)
		video_dconv3 = tf.nn.relu(video_dconv3)

		video_dconv4 = deconv3d(video_dconv3,[4,4,4,64,128],[batch_size, 16, 32, 32, 64],[1,2,2,2,1],'SAME',name='video_deconv4')
		video_dconv4 = batch_norm2(video_dconv4,scope='vd_bn_dconv4',num_of_filters=64)
		video_dconv4 = tf.nn.relu(video_dconv4)

		video_dconv5 = deconv3d(video_dconv4,[4,4,4,3,64],[batch_size, 32, 64, 64, 3],[1,2,2,2,1],'SAME',name='video_deconv5')
		video_dconv5 = tf.nn.tanh(video_dconv5)

		# mast out... (for the mast net)

		mask_deconv5, mask_deconv5_weights = deconv3d(video_dconv4,[4,4,4,1,64], [batch_size, 32, 64, 64, 1],[1,2,2,2,1],'SAME',name='mask_deconv5',with_w=True)
		mask_deconv5 = tf.nn.sigmoid(mask_deconv5)

		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1, scope='mask_l1')
		mask_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [mask_deconv5_weights])

		return video_dconv5,mask_deconv5,mask_penalty

def discriminator_net(input,reuse=None):
		disc_conv1 = conv3d(input,[4,4,4,3,64],[1,2,2,2,1],'SAME',name='dc_conv1',reuse=reuse)
		disc_conv1 = lrelu(disc_conv1)

		disc_conv2 = conv3d(disc_conv1,[4,4,4,64,128],[1,2,2,2,1],'SAME',name='dc_conv2',reuse=reuse)
		disc_conv2 = batch_norm2(disc_conv2,scope='dc_bn_conv2',num_of_filters=128,eps=1e-3,reuse=reuse)
		disc_conv2 = lrelu(disc_conv2)

		disc_conv3 = conv3d(disc_conv2,[4,4,4,128,256],[1,2,2,2,1],'SAME',name='dc_conv3',reuse=reuse)
		disc_conv2 = batch_norm2(disc_conv3,scope='dc_bn_conv3',num_of_filters=256,eps=1e-3,reuse=reuse)
		disc_conv3 = lrelu(disc_conv3)

		disc_conv4 = conv3d(disc_conv3,[4,4,4,256,512],[1,2,2,2,1],'SAME',name='dc_conv4',reuse=reuse)
		disc_conv4 = batch_norm2(disc_conv4,scope='dc_bn_conv4',num_of_filters=512,eps=1e-3,reuse=reuse)
		disc_conv4 = lrelu(disc_conv4)

		disc_conv5 = conv3d(disc_conv4,[2,4,4,512,2],[1,1,1,1,1],'VALID',name='dc_conv5',reuse=reuse)
		final = tf.reshape(disc_conv5[:,:,:,:,0],[batch_size])

		return tf.nn.sigmoid(final),final

def gen_video(z):
	background = static_net(z)
	background = tf.tile(tf.reshape(background,[batch_size,1,64,64,3]),[1,32,1,1,1]) #replicate frames to change bg to [batchsize,32,64,64,3]

	foreground,mask,penalty = video_net_and_mask(z)
	mask = tf.tile(mask,[1,1,1,1,3])#replicate frames to change mask to [batchsize,32,64,64,3]

	video = foreground*mask + background*(1-mask) #dot multiply
	return video,penalty

#load parameters from torch7 file
def load(sess,param):
	if param is not None:
		data_dict = param
		for key in data_dict:
			with tf.variable_scope(key, reuse=True):
				for subkey in data_dict[key]:
					 try:
					 	var = tf.get_variable(subkey)
					 	sess.run(var.assign(data_dict[key][subkey]))
					 	print 'Assign pretrain model ' + subkey + ' to ' + key
					 except:
					 	print 'Ignore ' + key

z = tf.placeholder(tf.float32,[batch_size,z_size],name='z')
x = tf.placeholder(tf.float32,[batch_size,32,64,64,3],name='x_real')

gen_videos,penalty = gen_video(z)

gen_D, gen_D_logits = discriminator_net(gen_videos)
real_D, real_D_logits = discriminator_net(x,reuse=True)

d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels=tf.ones_like(real_D_logits)))

d_loss_gen = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_D_logits, labels=tf.zeros_like(gen_D_logits)))

g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_D_logits, labels=tf.ones_like(gen_D_logits)))

with tf.name_scope("d_loss"):
	d_loss = d_loss_gen + d_loss_real
	tf.summary.scalar("d_loss",d_loss)
with tf.name_scope("g_loss"):
	g_loss = g_loss + penalty
	tf.summary.scalar("g_loss",g_loss)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dc_' in var.name]
g_vars = [var for var in t_vars if var not in d_vars]

# auto adjusting learning rate
#learning_rate = tf.train.exponential_decay(starter_learning_rate, train_iters, 10000, 0.9, staircase=True)
d_optim = tf.train.AdamOptimizer(starter_learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(starter_learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

# load data
if mode == "train":
	print "Start loading data....."
	hf = h5py.File(dataset_filename,'r')
	caps = hf.get('train_img')
	total_count = caps.shape[0]
	train_data = hf.get('train_img')[:total_count].reshape(-1,C, A*B*channel)
	print "Data loaded"

fetches = []
fetches.extend([gen_D,real_D,d_loss,g_loss,d_optim,g_optim])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.InteractiveSession(config=config)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
#should init before restore
sess.run(init)

print "start to load pretrained model..."
G_name = './beach_G.npy'
param_G = np.load(G_name).item()
D_name = './beach_D.npy'
param_D = np.load(D_name).item()

load(sess,param_G)
load(sess,param_D)

#restore the pretrained model with beach0 dataset
if restore == True:
	saver = tf.train.Saver(var_list=t_vars, max_to_keep=50) #max_to_keep : maximum number of recent checkpoint to keep,default to 5
	saver.restore(sess, restore_path)

merged = tf.summary.merge_all()
writer=tf.summary.FileWriter(log_path)

if mode == "train":
	for i in range(train_iters):
		xtrain = next_batch(train_data)
		xtrain = xtrain.reshape(-1,C,A,B,channel)
		z_sample = np.random.normal(1.0,size=(batch_size,z_size)) 
		feed_dict={x:xtrain,z:z_sample}

		results = sess.run(fetches,feed_dict)
		gen_D,real_D,d_loss,g_loss,_,_=results

		rf = sess.run(merged,feed_dict=feed_dict)
		writer.add_summary(rf,i)
		writer.flush()
		
		if i%10 == 0:
			print ("iter=%d : D_Loss: %f G_Loss: %f" % (i, d_loss, g_loss))

		if (i+1)%50==0:
			ckpt_file=model_file_name+"beach__"+str(i)+".ckpt"
			print ("Model saved in file: %s" % saver.save(sess,ckpt_file))

		if (i+1)%50==0:
			sample = sess.run(gen_videos,{z:z_sample})
			output = image_file_name + "__" + str(i)
			np.save(output,(sample*0.5+0.5)*255)
			print("Outputs image saved in file: %s" % output)

	writer.close()
	sess.close()

if mode == "generate":
	z_sample = np.random.normal(1.0,size=(batch_size,z_size))
	sample = sess.run(gen_videos,{z:z_sample})
			
	output = "./pretrain/normal_0.0_1.0"
	np.save(output,(sample*0.5+0.5)*255)
	print("Outputs image saved in file: %s" % output)