#!/usr/bin/env python
# coding: utf-8

# ### Pre-Import

# In[1]:


get_ipython().system('pip install "git+https://github.com/salaniz/pycocoevalcap.git"')


# ### Imports

# In[2]:


from tqdm.notebook import tqdm
import time
import os
import json
import requests
from struct import unpack

import h5py
import pickle
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow_addons.rnn import LayerNormSimpleRNNCell

import keras
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Lambda, Input, LSTM, GRU, RNN, Embedding, Multiply,  Concatenate, TimeDistributed, Dense, Bidirectional, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization

TRAIN_DATA_FILENAME = "../input/project-data/eee443_project_dataset_train.h5"
TRAIN_IMAGES_DIRECTORY = "../input/images-for-train/train_images/"
TRAIN_FEATURES_DIRECTORY = "../input/tupled-data/train_tupled_data"

TEST_DATA_FILENAME = "../input/project-data/eee443_project_dataset_test.h5"
TEST_IMAGES_DIRECTORY = "../input/images-for-test/test_images/"
TEST_FEATURES_DIRECTORY = "../input/tupled-data/test_tupled_data"


# ### Downloading and Cleaning Data

# In[3]:


def save_ims(data, pathname):
    """
    A function which saves images from given urls
    :param data: the array which holds url values
    :param pathname: the save path
    """
    s = time.time()
    i = 0

    if not os.path.exists(pathname):
        os.makedirs(pathname)

    for url in data:

        url = url.decode()
        name = url.split("/")[-1].strip()
        path = os.path.join(pathname, name)

        if not os.path.exists(path):

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
            response = requests.get(url, stream=True, headers=headers)

            # check if data is obttained successfully
            if response.status_code == 200:
                    with open(path, 'wb') as outfile:
                        outfile.write(response.content)

        # prints affirmation at each 1000 iterations
        if i % 1000 == 1:

            p = time.time() - s
            it = p/i
            print("{:.2f} mins passed. {:.2f} seconds per iter. Iteration {}".format(p/60, it, i))

        i += 1


class JPEG:
    """
    The JPEG class which helps in cleaning the data, more info can be found on the report.
    """
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break



def clear_bad_JPEG(root_img):
    """
    A funciton which clears a directory from bad JPEG's (.jpg files that are not actually images)
    :param root_img: tthe directory of jpg images
    """

    images = os.listdir(root_img)

    bads = []

    for img in tqdm(images):
        image = root_img + img
        image = JPEG(image)
        try:
            image.decode()
        except:
            bads.append(img)
    print(bads)
    for name in bads:
        os.remove(root_img + name)


# # Data Processing

# In[4]:


def caption_array_to_str(caption_array):
    """
    A function which formats a caption numpy array to a list of string(s). Used for evaluation and prediction.
    :param caption_array: The numpy array which stores captions/predicted captions
    :return: a list of strings that contain the caption
    """
    
    list_of_captions = []

    caption = ""

    if(caption_array.ndim == 1):
        caption_array = np.expand_dims(caption_array, axis=0)

    for caps in caption_array:

        for word in caps:

            if (word == 'x_NULL_') or (word == 'x_START_') or (word == 'x_END_'):
                continue
                
            caption += word + " "
            
            
        list_of_captions.append(caption.strip())
        caption = ""

        
    return list_of_captions


def get_caption(name_list, imid, cap, name):
    """
    A function which returns the caption of an image given its name
    :param name_list:
    :param imid: image id vector
    :param cap: cap array (holds all captions)
    :param name: the name of the image
    :return: the caption numpy array
    """
  
    ind = name_list.index(name) + imid.min()
  
    return cap[np.where(imid == ind)]


def create_pre_processed_set(image_directory, shuffle=False):
    """
    A function which creates a set of preprocessed images ready for feature extraction.
    :param image_directory: The directory containing all images
    :param shuffle: whetther we want to shuffle data or not
    :return: a tf.data.Dataset that contains all preprocessed images (contains meaning it contains the formula to create them,
    however the data is not actual loaded into memory until called)
    """

    def process_files(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img

    def process_name(path):
        name = path.numpy().decode().split("/")[-1]
        return name

    def process(path):
        name = tf.py_function(process_name, [path], tf.string)
        img = tf.py_function(process_files, [path], tf.float32)
        return (img, name)
    
    file_data = tf.data.Dataset.list_files(str(image_directory) + "*.jpg", shuffle=shuffle)
    
    return file_data.map(lambda x: process(x))


def create_features(filename, images, name_list, imid, cap, process_size = 250):
    """

    :param filename: the filename for the data to be dumped in
    :param images: the image tf.data.Dataset (created in the previous function)
    :param name_list: a list of image url names
    :param imid: image id array
    :param cap: captions array
    :param process_size: the batch size with which the feattures are extracted
    :return: the length of the data
    """

    inception = tf.keras.applications.InceptionV3(weights='imagenet')
    inception = tf.keras.Model(inception.input, inception.layers[-2].output)
    
    
    length = 0
    
    with open(filename, "wb") as outfile:
        
        for data in tqdm(images.batch(process_size)):
                
            image = data[0]
            name = data[1].numpy()
            feature = inception(image).numpy()
            
            for i in range(feature.shape[0]):
                
                f = feature[i].squeeze()
                n = name[i].decode()
                c = get_caption(name_list, imid, cap, n)
                
                tp = (f, c, n)
                pickle.dump(tp, outfile)
                
                length += 1

    outfile.close()
    return length
    
    
def loadpickle(filename):
    """
    A generator function which yields data in a file until there is none
    :param filename:
    :return: loaded tuple (or any other data that is stored)
    """

    with open(filename, "rb") as f:

        while True:

            try:
                yield pickle.load(f)

            except EOFError:
                break
            

                
def create_data(feature_directory, url=None, imid=None, cap=None, image_directory=None):
    """

    :param feature_directory: directory of the pickled data
    :param url: url array (from the given dataset)
    :param imid: image id vector (from the given dataset)
    :param cap: cap array (from the given dataset)
    :param image_directory: the directory images are stored in
    :return: the dataset and the length of said dataset
    """
    
    length = -1
    
    if not os.path.isfile(feature_directory):
        
        if not image_directory:
            raise Exception("No image directory given. Enter image directory for feature extraction.")
        
        name_list = [u.split("/")[-1].strip() for u in np.char.decode(url).tolist()]
        images = create_pre_processed_set(image_directory)
        length = create_features(feature_directory, images, name_list, imid, cap)
        
    dataset = tf.data.Dataset.from_generator(loadpickle, args=[feature_directory], output_types=(np.float32,np.int32, tf.string))

    if length == -1:
        length = dataset.reduce(0, lambda x, _: x + 1).numpy()
    
    return dataset, length


#  ### Get Train Image Dataset

# In[5]:


f = h5py.File(TRAIN_DATA_FILENAME, "r")

for key in list(f.keys()):
    print(key, ":", f[key][()].shape)

train_cap = f["train_cap"][()]
train_imid = f["train_imid"][()]
train_url = f["train_url"][()]
word_code = f["word_code"][()]


df = pd.DataFrame(word_code)
df = df.sort_values(0, axis=1)
words = np.asarray(df.columns)

wordtoix = {}
for i in range(len(words)):
  word = words[i]
  wordtoix[word] = i


train_data, train_data_length = create_data( TRAIN_FEATURES_DIRECTORY, train_url, train_imid, train_cap, TRAIN_IMAGES_DIRECTORY)

print("Vocab Size =", len(words))
print( "{} of {} retrieved. {:.1f}% of data is clean.".format(train_data_length, len(train_url), 100 * train_data_length/len(train_url) ) )

# delete after use so that memory is not loaded
del train_cap 
del train_imid
del train_url 
del word_code


# In[6]:


for d in train_data.shuffle(1000).take(1):
    features = d[0]
    image_name= d[2].numpy().decode()
    captions = d[1].numpy()
    
    
    im = cv2.imread(TRAIN_IMAGES_DIRECTORY + image_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
    cap = caption_array_to_str(words[captions])

    for c in cap:
        print(c)
        
    print(features.shape, captions.shape)


#  ### Get Test Image Dataset

# In[7]:


f = h5py.File(TEST_DATA_FILENAME, "r")

for key in list(f.keys()):
    print(key, ":", f[key][()].shape)

test_cap = f["test_caps"][()]
test_imid = f["test_imid"][()]
test_url = f["test_url"][()]


test_data, test_data_length = create_data(TEST_FEATURES_DIRECTORY, test_url, test_imid, test_cap, TEST_IMAGES_DIRECTORY)

print( "{} of {} retrieved. {:.1f}% of data is clean.".format(test_data_length, len(test_url), 100 * test_data_length/len(test_url) ))

# delete after use so that memory is not loaded
del test_cap 
del test_imid
del test_url


# In[8]:


for d in test_data.shuffle(1000).take(1):
    features = d[0]
    captions = d[1].numpy()
    image_name= d[2].numpy().decode()
    
    im = cv2.imread(TEST_IMAGES_DIRECTORY + image_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
    cap = caption_array_to_str(words[captions])
    
    for c in cap:
        print(c)


# # Model Preparation

# In[9]:


def data_generator(dataset, max_length, num_photos_per_batch, vocab_size):
    """
    The data generatoor function which generates training data
    :param dataset: the tf.data.Dataset object containing tthe tuple (feature, captions, name)
    :param max_length: maximum sentence/caption length in terms of words
    :param num_photos_per_batch: batch size
    :param vocab_size: vocabulary size, word count
    :return: the training data to be used at every iteration
    """

    X1, X2, y = [], [], []
    i = 0

    while True:

        for data in dataset:

            i += 1
            feature = data[0].numpy()
            caps = data[1].numpy()

            for j in range(caps.shape[0]):

                seq = caps[j]

                for k in range(1, seq.shape[0]):

                    in_seq = pad_sequences([seq[:k]], maxlen=max_length)[0]
                    out_seq = to_categorical([seq[k]], num_classes=vocab_size)[0]

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)


            if i == num_photos_per_batch:

                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = [], [], []
                i = 0

    yield [np.array(X1), np.array(X2)], np.array(y)

        
def create_embedding(wordtoix):
    """
    Mehod which creates the embedding matrix wih a given word index dictionary
    :param wordtoix: word intex
    :return: the embedding matrtix
    """
    # Load Glove vectors
    glove_dir = '../input/glove6b200d/glove.6B.200d.txt'
    embeddings_index = {} # empty dictionary
    f = open(glove_dir, encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # if (word == 'startseq' or word == 'unk' ):
        #   print(word)

        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))


    embedding_dim = 200
    vocab_size = 1004

    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():

        if (word == 'x_UNK_'):
          word = 'unk'

        embedding_vector = embeddings_index.get(word)
        if  embedding_vector is None:
          print(word)

        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


# ### 1st Iteration Models

# In[10]:


def create_merge_model(embedding_matrix):
    
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(256, activation='relu')(img1)
    
    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)
    seq3 = LSTM(256)(seq2)
    
    #add, not concatenate! wrong 
    dec1 = add([img1, seq3])
    dec2 = Dense(256, activation='relu')(dec1)
    outputs = Dense(vocab_size, activation='softmax')(dec2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    #set embedding layer's weight matrix 
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = True
    
    return model


def create_init_inject_model(embedding_matrix):
    
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(256, activation='relu')(img1)
    
    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)

    #image is set as state 
    seq3,state = GRU(256,return_state = True)(seq2,initial_state = img1)
    
    dec2 = Dense(256, activation='relu')(seq3)
    outputs = Dense(vocab_size, activation='softmax')(dec2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)    
    
    #set embedding layer's weight matrix 
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = True
    
    return model



def create_pre_inject_model(embedding_matrix):
    
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(embedding_dim, activation='relu')(img1)
    img2_reshaped = Reshape((1, embedding_dim), input_shape=(embedding_dim,))(img2)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)
    seq3,state3 = GRU(256,return_state = True)(img2_reshaped)
    seq4,state4 = GRU(256,return_state = True)(seq2, initial_state = state3)
    dec = Dense(256, activation='relu')(seq4)
    outputs = Dense(vocab_size, activation='softmax')(dec)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)    

    model.layers[4].set_weights([embedding_matrix])
    model.layers[4].trainable = True
   
        
    return model



def create_par_inject_model(embedding_matrix):
    
    max_length = 17
    vocab_size = 1004
    embedding_dim = 200
    
    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(200, activation='relu')(img1)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)

    mul = Multiply()([img2,seq2])

    seq3 = LSTM(256)(mul)
    dec = Dense(256, activation='relu')(seq3)
    outputs = Dense(vocab_size, activation='softmax')(dec)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.layers[3].set_weights([embedding_matrix])
    model.layers[3].trainable = True
    
    return model


# ### Models with Layer Normalization

# In[11]:


def create_merge_model_best(embedding_matrix):

    inputs1 = Input(shape=(2048,))
    img = Dense(128, activation='relu', kernel_initializer='random_normal')(inputs1)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)

    seq2 = RNN(LayerNormLSTMCell(128))(seq1)
    seq3 = Dropout(0.5)(seq2)

    dec1 = add([img, seq3])
    dec2 = Dense(128, activation='relu',kernel_initializer='random_normal')(dec1)
    outputs = Dense(vocab_size, activation='softmax',kernel_initializer='random_normal')(dec2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    #set embedding layer's weight matrix 
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = True

    return model




def create_par_inject_model_best(embedding_matrix):

    inputs1 = Input(shape=(2048,))
    img1 = Dropout(0.5)(inputs1)
    img2 = Dense(200, activation='relu',kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-8))(img1)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)

    mul = Multiply()([img1,seq2])

    seq3 = RNN(LayerNormLSTMCell(256))(mul)
    seq4 = Dropout(0.5)(seq3)
    dec = Dense(256, activation='relu',kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-8))(seq4)
    outputs = Dense(vocab_size, activation='softmax',kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-8))(dec)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    model.layers[3].set_weights([embedding_matrix])
    model.layers[3].trainable = True

    return model




def create_pre_inject_model_best(embedding_matrix):

    inputs1 = Input(shape=(2048,))
    img = Dense(embedding_dim, activation='relu',kernel_initializer='random_normal')(inputs1)
    img_reshaped = Reshape((1, embedding_dim), input_shape=(embedding_dim,))(img)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)


    seq3,state3 = RNN(LayerNormSimpleRNNCell(512),return_state = True)(img_reshaped)
    seq4,state4 = RNN(LayerNormSimpleRNNCell(512),return_state = True)(seq2, initial_state = state3)

    seq5 = Dropout(0.5)(seq4)
    dec = Dense(512, activation='relu',kernel_initializer='random_normal')(seq5)
    outputs = Dense(vocab_size, activation='softmax',kernel_initializer='random_normal')(dec)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.layers[3].set_weights([embedding_matrix])
    model.layers[3].trainable = True


    return model



def create_init_inject_model_best(embedding_matrix):

    inputs1 = Input(shape=(2048,))

    img = Dense(512, activation='relu')(inputs1)

    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    seq2 = Dropout(0.5)(seq1)

    #image is set as state 
    seq3,state = RNN(LayerNormSimpleRNNCell(512),return_state = True)(seq2,initial_state = img)
    seq4 = Dropout(0.5)(seq3)

    dec = Dense(512, activation='relu')(seq4)
    outputs = Dense(vocab_size, activation='softmax')(dec)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)


    #set embedding layer's weight matrix 
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = True

    return model


# ### Bi-Directional Model

# In[12]:


def create_bilstm_model_layernorm(embedding_matrix):

    inputs1 = Input(shape=(2048,))
    img1 = Dense(embedding_dim, input_shape=(2048,), activation='relu')(inputs1)
    img2 = RepeatVector(max_length)(img1)
     
    inputs2 = Input(shape=(max_length,))
    seq1 = Embedding(vocab_size,embedding_dim, input_length=max_length, mask_zero=True)(inputs2)
    seq2 = Bidirectional(RNN(LayerNormLSTMCell(256), return_sequences = True))(seq1)
    seq3 = Dropout(0.5)(seq2)
    seq4 = TimeDistributed(Dense(embedding_dim))(seq3)

    comb1 = Concatenate(axis=2)([img2, seq4])
    comb2 = Dropout(0.5)(comb1)
    comb3 = Bidirectional(RNN(LayerNormLSTMCell(1000), return_sequences = False))(comb2)
    comb4 = Dense(vocab_size, activation = 'softmax')(comb3)
        
    model = Model(inputs=[inputs1, inputs2], outputs=comb4)
    model.summary()
    
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = True
      
    return model


# # Train (skip to load pre-existing models)

# ### Creating the Embedding Matrix

# In[13]:


print("Creating embedding matrix...")
embedding_matrix = create_embedding(wordtoix)
print('Embedding matrix is ready!')


# ### Start Training

# In[14]:


data_length = 10000 #train_datta_length
train_data = train_data.take(data_length)

val_length = round(data_length * 0.15)
train_length = data_length - val_length

val_dataset = train_data.take(val_length) 
train_dataset = train_data.skip(val_length)


max_length = 17
vocab_size = 1004
embedding_dim = 200

epochs = 12


# In[15]:


train_ = False

if train_ == True: 
    model_names = ["create_merge_model_best", 
                   "create_par_inject_model_best", 
                   "create_pre_inject_model_best", 
                   "create_init_inject_model_best"]

    history_list = {}

    for i in range(2, 3):

        if i == 0:
            model = create_merge_model_best(embedding_matrix) 
            batch_size = 128
        if i == 1:
            model = create_par_inject_model_best(embedding_matrix) 
            batch_size = 64
        if i == 2:
            model = create_pre_inject_model_best(embedding_matrix) 
            batch_size = 32
        if i == 3:
            model = create_init_inject_model_best(embedding_matrix) 
            batch_size = 128


        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print("Model compiled...")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_names[i], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        print("Creating Generators")
        train_generator = data_generator(train_dataset, max_length, batch_size, vocab_size)
        val_generator = data_generator(val_dataset, max_length, batch_size, vocab_size)

        train_step = train_length // batch_size
        val_step = val_length // batch_size


        print("Starting training...")
        start = time.time()

        history = model.fit(train_generator,
                        steps_per_epoch = train_step,
                        validation_data = val_generator,
                        validation_steps = val_step,
                        epochs = epochs,
                        callbacks = callbacks_list,
                            workers = 0)

        history_list[model_names[i]] = history.history


        model.save(model_names[i] + "_save")


        print("Time spent {:.2f} mins.".format( (time.time()-start)/60 ))


    with open('loss_history.json', 'w') as fp:
        json.dump(history_list, fp)


# ### Load Pre-Saved Models and History Dictionaries

# In[16]:


models = {}

path = "../input/models/old_models/old_models/"

with open(path + 'old_loss_history.json', 'r') as fp:
    history_dict = json.load(fp)
    
with open("../input/models/bidirectional/loss_history_alt_batchnorm.json", 'r') as fp:
    bihist = json.load(fp)
    

for key in history_dict:
    try:
        models[key] = tf.keras.models.load_model(path + key, custom_objects={'LayerNormLSTMCell':LayerNormLSTMCell})
    except:
        try:
            models[key] = tf.keras.models.load_model(path + key, custom_objects={'LayerNormSimpleRNNCell':LayerNormSimpleRNNCell})
        except:
            print("Could not load model.")
            
            
try:
    bi = tf.keras.models.load_model("../input/models/bidirectional/create_bilstm_model_layernorm_batchnorm", custom_objects={'LayerNormLSTMCell':LayerNormLSTMCell})
except:
    try:
        bi= tf.keras.models.load_model("../input/models/bidirectional/create_bilstm_model_layernorm_batchnorm", custom_objects={'LayerNormSimpleRNNCell':LayerNormSimpleRNNCell})
    except:
        print("Could not load model.")


# ### Plot History (Different Blocks for the Report)

# In[17]:


def plot_loss(model_name, hist):
    
    fig = plt.figure(figsize=(12, 8), dpi=160, facecolor='w', edgecolor='k')
    fig.suptitle(model_name, fontsize=13)
    plt.plot(hist["loss"], "C2", label="Train Sequential Cross Entropy Loss")
    plt.plot(hist["val_loss"], "C3", label="Validation Sequential Cross Entropy Loss")
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("pre_inject.png", bbox_inches='tight')


# In[18]:


fig = plt.figure(figsize=(12, 4), dpi=160, facecolor='w', edgecolor='k')
c = ["C2", "C3"]

plt.subplot(1,2,1)
h1 = list(history_dict.values())[4]
h2 = list(bihist.values())[0]

plt.plot(h1["loss"], c[0], label="Modified Merge Model")
plt.plot(h2["loss"], c[1], label="Bi-Directional Model")


plt.title("Train Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')   
plt.legend() 


plt.subplot(1,2,2)

plt.plot(h1["val_loss"], c[0], label="Modified Merge Model")
plt.plot(h2["val_loss"], c[1], label="Bi-Directional Model")


plt.title("Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')   
plt.legend()  
plt.savefig("Loss_Plots_BiDir_Merge.png", bbox_inches='tight')


# In[19]:


fig = plt.figure(figsize=(6, 4), dpi=160, facecolor='w', edgecolor='k')

c = ["C1", "C2", "C3", "C4"]
names = ["Merge Model", "Par Inject Model", "Pre Inject Model", "Init Inject Model"]
# names = ["Modified Merge Model", "Modified Par Inject Model",
#              "Modified Pre Inject Model", "Modified Init Inject Model"]
h = list(history_dict.values())[4:]
fig = plt.figure(figsize=(12, 4), dpi=160, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
for i in range(4):
    plt.plot(h[i]["loss"], c[i], label=names[i])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')   
plt.legend()    


plt.subplot(1,2,2)
for i in range(4):
    plt.plot(h[i]["val_loss"], c[i], label=names[i])
plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')   
plt.legend()

plt.savefig("Loss_Plots.png", bbox_inches='tight')


# In[20]:


fig = plt.figure(figsize=(18, 12), dpi=160, facecolor='w', edgecolor='k')
c = ["C2", "C3"]
names = ["Merge Model", "Par Inject Model", "Pre Inject Model", "Init Inject Model",
            "Modified Merge Model", "Modified Par Inject Model",
             "Modified Pre Inject Model", "Modified Init Inject Model"]


for i in range(4):
    plt.subplot(2,2,i + 1)
    h1 = list(history_dict.values())[i]
    h2 = list(history_dict.values())[i + 4]

    plt.plot(h1["loss"], c[0], label=names[i])
    plt.plot(h2["loss"], c[1], label=names[i + 4])
    
    
    plt.title('Train Loss for ' + names[i] + "s")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')   
    plt.legend()  
    
plt.savefig("Loss_Plots_Comparison_Modified.png", bbox_inches='tight')


# # Evaluation

# In[21]:


def create_evaluation_dictionary(model, dataset, data_length, batch_size, words):
    """
    Creates a dictionary for the score listing function
    :param model: the model the predictions will be based on
    :param dataset: the tf.data.Dataset object, in our case this is the test dataset
    :param data_length: the data length of the dataset
    :param batch_size: this is the process size, how many predictions are to be done at once
    :param words: the words dictionary
    :return:
    """
        
    pred_container = {}
    actual_container = {}

    start = time.time()
    
    referenced = dataset.map(lambda x, y, z: (x, z))
    
    for batch in referenced.batch(batch_size):
        
        feats = batch[0].numpy()
        iteration = feats.shape[0]
        seq = np.tile(np.array([1] + [0]*16), (iteration, 1))
        
        for i in range(16):

            pred = model.predict([feats, seq])
            seq[:, i+1] = np.argmax(pred, axis=1)
            
        for i in range(iteration):
            
            name = batch[1].numpy()[i].decode()
            pred_container[name] = caption_array_to_str(words[seq[i]])

    for d in dataset.take(data_length):
        name = d[2].numpy().decode()
        caption = d[1].numpy()
        actual_container[name] = caption_array_to_str(words[caption])

        
    return actual_container, pred_container  



from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider 
from pycocoevalcap.meteor.meteor import Meteor

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")]
    
    final_scores = {}
    
    for scorer, method in scorers:
        
        score, scores = scorer.compute_score(ref, hypo)
        
        if type(score) == list:
            
            for m, s in zip(method, score):
                
                final_scores[m] = s
                
        else:
            
            final_scores[method] = score
            
            
    return final_scores 


def print_metrics(actual, preds, model_name):
    
    metric_dict = score(actual, preds)
    
    b1 = metric_dict["Bleu_1"] * 100
    b2 = metric_dict["Bleu_2"] * 100
    b3 = metric_dict["Bleu_3"] * 100
    b4 = metric_dict["Bleu_4"] * 100
    m = metric_dict["METEOR"] * 100
    r = metric_dict["ROUGE_L"] * 100
    c = metric_dict["CIDEr"] * 100
    string = "\n--------------------\nModel: {}\n--------------------\nBLEU-1: {:.1f}\nBLEU-2: {:.1f}\nBLEU-3: {:.1f}\nBLEU-4: {:.1f}\nMETEOR:  {:.1f}\nROGUE_L: {:.1f}\nCIDEr: {:.1f}\n".format(model_name, b1, b2, b3, b4, m, r, c)
    print(string)


# ### Evaluate Bleu, Meteor, CIDEr and Rouge_L Scores

# In[22]:


print("Example of a score output (this might take a short while)...\n")
data_size = test_data_length
process_size = 250
prediction_data = test_data.take(data_size)

actual, preds = create_evaluation_dictionary(models["create_pre_inject_model_best"], prediction_data, data_size, process_size, words)
print_metrics(actual, preds, key)


# ### Display Predicted Image with Caption (for the report)

# In[23]:


def predict_for_extra(model, feature_model, img_path):
    """
    prediction function for extra images, not provided to us via test dataset
    this is for fun but also to see if the model can actually correctly guess any other image (spoiler: it can)
    :param model: the prediction model
    :param feature_model: feature extraction model
    :param img_path: path of image to be predicted
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    img = feature_model(img)
    img = img.numpy().reshape(1, -1)

    seq = np.array([0]*17).reshape(1, -1)

    seq[:, 0] = 1

    for i in range(16):
        pred = model.predict([img,seq])
        seq[:, i+1] = np.argmax(pred)

    seq = seq.reshape(-1)

    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig = plt.figure(dpi=160, facecolor='w', edgecolor='k')
    plt.imshow(im)
    plt.title("Prediction: " + caption_array_to_str(words[seq])[0], fontsize=9)
    plt.axis('off')

    plt.show()
    
    

def predict_caption(model, image_directory, feature_model, amount, dr):
    """
    A function which predicts and saves many caption/prediction duos plotted with their respective images
    :param model: prediction model
    :param image_directory: the image directory, test images directory
    :param feature_model: feature extraction model
    :param amount: how many pictures the function will be predicting
    :param dr: the save directory
    """
    
    for d in test_data.shuffle(test_data_length).take(amount):
    
        img_path = str(image_directory + d[2].numpy().decode())
        c = d[1].numpy()
        k = np.random.randint(c.shape[0])
    
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        img = feature_model(img)
        img = img.numpy().reshape(1, -1)

        seq = np.array([0]*17).reshape(1, -1)

        seq[:, 0] = 1

        for i in range(16):
            pred = model.predict([img,seq])
            seq[:, i+1] = np.argmax(pred)

        seq = seq.reshape(-1)

        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(dpi=160, facecolor='w', edgecolor='k')
        plt.imshow(im)
        plt.title("Prediction: " + caption_array_to_str(words[seq])[0] + "\nActual Caption: " + caption_array_to_str(words[c[k]])[0], fontsize=9)
        plt.axis('off')
        plt.show()
        # plt.savefig(dr + img_path.split("/")[-1], bbox_inches='tight')
        plt.close("all")
 


# In[24]:


inception = tf.keras.applications.InceptionV3(weights='imagenet')
inception = tf.keras.Model(inception.input, inception.layers[-2].output)


# In[25]:


best_model = models["create_pre_inject_model"] #change this to whatever model you want
predict_caption(best_model, "../input/images-for-test/test_images/", inception, 3, "")

