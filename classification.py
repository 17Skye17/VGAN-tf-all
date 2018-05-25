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
mode = "test"
restore = True

model_file_name = "./models-399after/"
dataset_filename = './ucf101.hdf5'
restore_path = './models/beach__249.ckpt'
log_path = "./logs/399/"
D_name = './classification_D.npy'
###################################################################

if os.path.exists(model_file_name) == False:
	os.mkdir(model_file_name)
if os.path.exists(log_path) == False:
	os.mkdir(log_path)
	
tf.reset_default_graph()

def next_batch(image,label):
    length=image.shape[0] #assuming the data array to be a np arry
    permutations=np.random.permutation(length)
    idxs=permutations[0:batch_size]
    imagebatch=np.zeros([batch_size, C, img_size*channel], dtype=np.float32)
    labelbatch = np.zeros([batch_size,C],dtype=np.float32)
    for i in range(len(idxs)):
        imagebatch[i,:]=image[idxs[i]].reshape(C,img_size*channel)
	labelbatch[i,:]=label[idxs[i]].reshape(C)
    return imagebatch,labelbatch

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

		disc_conv5 = tf.nn.dropout(disc_conv4,0.5)
		disc_conv5 = conv3d(disc_conv5,[2,4,4,512,101],[1,1,1,1,1],'VALID',name='dc_conv5',reuse=reuse)

		return tf.nn.softmax(disc_conv5),disc_conv5

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

def accuracy(logit,labels):
	correct_pred = tf.equal(tf.argmax(logit,1),tf.cast(labels,tf.int64))
	acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	return acc

sequence = tf.placeholder(tf.float32,[batch_size,C,A,B,channel],name='input')
label = tf.placeholder(tf.float32,[batch_size,C],name="label") #label[:,0]=[bs,1]
D_prob, D_logits = discriminator_net(sequence) 

D_logits = tf.reshape(D_logits,[batch_size,101]) #[bs,101]
labels = tf.one_hot(tf.cast(label[:,0],tf.int64),101,dtype=tf.float32)
with tf.name_scope("accuracy"):
	accur = accuracy(D_logits,label[:,0])
	tf.summary.scalar("accuracy",accur)
	
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=D_logits))
	tf.summary.scalar("loss",loss)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if "dc_" in var.name]
# auto adjusting learning rate
#learning_rate = tf.train.exponential_decay(starter_learning_rate, train_iters, 10000, 0.9, staircase=True)
d_optim = tf.train.AdamOptimizer(starter_learning_rate, beta1=beta1).minimize(loss, var_list=d_vars)

# load data
if mode == "train":
	print "Start loading data....."
	hf = h5py.File(dataset_filename,'r')
	caps = hf.get('train_img')

	total_count = caps.shape[0]

	train_data = hf.get('train_img')[:total_count].reshape(-1,C, A*B*channel)
	train_label = hf.get('train_labels')[:total_count].reshape(-1,C)
	print "Train Data loaded"
	
	caps1 = hf.get('val_img')
	total_count1 = caps1.shape[0]
	val_data = hf.get('val_img')[:total_count1].reshape(-1,C,A*B*channel)
	val_label = hf.get('val_labels')[:total_count1].reshape(-1,C)
	print "Val Data loaded"
	
if mode == "test":
	print "Start loading data....."
	hf = h5py.File(dataset_filename,'r')
	caps = hf.get('test_img')
	total_count = caps.shape[0]
	test_data = hf.get('test_img')[:total_count].reshape(-1,C,A*B*channel)
	test_label = hf.get('test_labels')[:total_count].reshape(-1,C)
	print "Test Data loaded"	


fetches = []
fetches.extend([accur,D_prob,loss,d_optim])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.InteractiveSession(config=config)


init = tf.global_variables_initializer()
#should init before restore
sess.run(init)

#restore the pretrained model with beach0 dataset
if restore == True:
	saver = tf.train.Saver(var_list=t_vars, max_to_keep=50) #max_to_keep : maximum number of recent checkpoint to keep,default to 5
	saver.restore(sess, restore_path)
else:
	saver = tf.train.Saver()
	print "start to load pretrained model..."

	param_D = np.load(D_name).item()
	load(sess,param_D)
	
merged = tf.summary.merge_all()
writer=tf.summary.FileWriter(log_path)

if mode == "train":
	for i in range(train_iters):
		train_image,train_labels = next_batch(train_data,train_label)
		train_image = train_image.reshape(-1,C,A,B,channel)

		feed_dict={sequence:train_image,label:train_labels}

		results = sess.run(fetches,feed_dict)
		accur,D_prob,loss,_=results

		rf = sess.run(merged,feed_dict=feed_dict)
		writer.add_summary(rf,i)
		writer.flush()
		
		if i%10 == 0:
			print ("iter=%d : Loss: %f Accuracy: %f " % (i, loss,accur))

	#	if i%10==0:
			val_image,val_labels = next_batch(val_data,val_label)
			val_image = val_image.reshape(-1,C,A,B,channel)
			feed_dict={sequence:val_image,label:val_labels}
			results = sess.run(fetches,feed_dict)
			accur,D_prob,loss,_=results
			print ("Val Loss: %f Accuracy: %f" % (loss,accur))
			
			ckpt_file=model_file_name+str(i)+".ckpt"
			print ("Model saved in file: %s" % saver.save(sess,ckpt_file))

if mode == "test":
	total_acc = 0.0
	for i in range(len(test_data)/batch_size):
		test_image,test_labels = next_batch(test_data,test_label)
		test_image = test_image.reshape(-1,C,A,B,channel)
		feed_dict={sequence:test_image,label:test_labels}

		results = sess.run(fetches,feed_dict)
		accur,D_prob,loss,_=results
		
		total_acc = total_acc + accur
		average_acc = total_acc/i
		print "accur = %f  average accur= %f"%(accur,average_acc)
writer.close()
sess.close()
