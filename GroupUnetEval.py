#Authors: DS291 and JH384

import numpy as np
from DiceScore import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import pickle
from tensorflow.keras import backend as K
import tensorflow as tf
tf.random.set_seed(1234)

################################################################### Loading Validation Data ###############################################################
#Method loads in the validation data.
def load_data(Data_path, Label_path, id_path):
    #Load validation IDs. Changing the final part can allow to test with Test data or even training.
    X_val = np.load(id_path + "X_val.npy")
    Y_val = np.load(id_path + "Y_val.npy")

    val_idx = X_val
    X_val = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 4, 240, 240,155)
    Y_val = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 240, 240,155)

    for i, ID in enumerate(val_idx):
        X_val[i,] = np.load(Data_path  + str(ID) + '_img.npy')
    X_val = X_val.reshape([-1,128,128,4])

    for i, ID in enumerate(val_idx):
        Y_val[i,] = np.load(Label_path + str(ID) + '_gt.npy')
    Y_val = Y_val.reshape([-1,128,128,4])

    return X_val, Y_val

################################################################### Output Images ###############################################################
#Puts one image over another
def overlay(slice1, slice2, index):
    #The ground truth slice should be the second slice
    plt.axis('off')
    plt.imshow(slice1, cmap=plt.cm.get_cmap('gray'))
    plt.savefig("brain.png", bbox_inches='tight')

    plt.axis('off')
    plt.imshow(slice2, cmap=plt.cm.get_cmap('gnuplot', 4))
    plt.savefig("seg.png", bbox_inches='tight')

    #After saving the images as .png, reopen as an image object to overlay 
    t1 = Image.open('brain.png')
    t2 = Image.open('seg.png')

    t1 = t1.convert("RGBA")
    t2 = t2.convert("RGBA")

    #Creates the blended image
    new_img = Image.blend(t1, t2, 0.5)
    new_img.save("./overlays/overlay" + str(index) + ".png","PNG")

def overlayKey(slice2):
    fig, ax = plt.subplots()
    plt.axis('off')
    cax = ax.imshow(slice2, cmap=plt.cm.get_cmap('gnuplot', 4))
    ax.set_title('Ground Truth')
    cbar = fig.colorbar(cax, ticks = [0.25,0.75,1.25,1.75], orientation = 'horizontal')
    cbar.ax.set_xticklabels(['Background','Necrotic Core', 'Edema', 'Enhancing Tumour'])
    plt.savefig("seg_key.png")

################################################################### Validation Data ###############################################################
Data_path = './dataset_conv/X/x'
Label_path = './dataset_conv/Y/y'
id_path = './dataset_conv/'
X_val, Y_val = load_data(Data_path, Label_path, id_path)

weights_file = open("class_weights.pkl", "rb")
weights = pickle.load(weights_file)
weights_file.close
weights_list = list(weights.values())

# Load in model:
unet_model = tf.keras.models.load_model('./models/group-UNet.h5',
                                            custom_objects={'lossFunc': weighted_dice(dice_loss_function, weights_list),
                                                            'dice_function': dice_function})


################################################################### Predictions ###############################################################
# Get predictions from validation data using model
val_Y_pre = np.argmax(unet_model.predict(X_val), axis=-1)
Y_val = np.argmax(Y_val, axis=-1)

#Flatten the predictions and labels for the classification report.
oneD_val_pre = K.flatten(val_Y_pre)
oneD_Y = K.flatten(Y_val)

classes = ['Background','Necrotic Core', 'Edema', 'Enhancing Tumour']
report = classification_report(oneD_Y, oneD_val_pre, target_names = classes)

# The prediction array and Y_val array are reshaped:
val_Y_pre = val_Y_pre.reshape(-1, 128, 128, 1)
Y_val_reshape = Y_val.reshape(-1, 128, 128, 1)

# The dice_function_loop is called to evaluate the predictions with the ground truth labels:
print("Dice scores using validation data: ", dice_function_loop(Y_val_reshape, val_Y_pre))
print(report)



################################################################### Image saves ###############################################################
# The following loop takes slices 1220, 1230 and saves their corresponding X data, Y data, and the predicted segmentation.
X_val = X_val.astype('uint8')
val_Y_pre = val_Y_pre.astype('uint8')
Y_val_reshape = Y_val_reshape.astype('uint8')

for i in range(1220,1230):
    overlay(X_val[i, :, :, 0], val_Y_pre[i, :, :, 0], (str(i) + '_Prediction'))
    overlay(X_val[i, :, :, 0], Y_val_reshape[i, :, :, 0], (str(i) + '_GroundTruth'))

overlayKey(val_Y_pre[1215, :, :, 0])