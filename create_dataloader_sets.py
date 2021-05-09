import sys, os
from tensorflow.keras.utils import to_categorical
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    ''' 
    Purpose: The Dataset Class inherits the Pytorch Dataset Class. It prepares the Dataset to be loaded by the Dataloader.
             Labels can either be binary or multiclass.
    Input:  Text Input: (input_ids, attention_mask, token_ids) from Transformer Encoder
            Image Indices: For loading the images
            Phase Name : "Train", "Validation", "Test"
    Output: X, img, y (where X is the text part, img is the image part and y is the label)
    '''
    def __init__(self, text_input, text_mask, token_id, img_indices, labels, phase):
        'Initialization'
        self.labels = labels
        self.text_input = text_input
        self.text_mask = text_mask
        self.text_token = token_id
        self.img_indices = img_indices
        self.phase = phase
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'validation': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.imgs = []   
        for i in range(len(self.labels)):
            #print(self.img_indices[i][0])
            img = self.read_image(self.img_indices[i][0])
            self.imgs.append(img)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
    
    def read_image(self, img_index):
#         print(img_index)
        i = img_index
        data_path = "./combined_image_dataset/"
        img_path = os.path.join(data_path, str(i)+'.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, str(i)+'.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(data_path, str(i)+'.jpeg')
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_path, str(i)+'.gif')
                    if not os.path.exists(img_path):
                        print(img_path)
                        raise FileNotFoundError 
        img = Image.open(img_path).convert('RGB')
        tsfm_img = self.data_transforms[self.phase](img) # 3x224x224
        
        return tsfm_img

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.text_token is not None:
            X = np.vstack((np.vstack((self.text_input[index], self.text_mask[index])), self.text_token[index]))
        else:
            X = np.vstack((np.vstack((self.text_input[index], self.text_mask[index]))))
        y = self.labels[index]
        
#         img = self.read_image(self.img_indices[index])
        img = self.imgs[index]

        return X, img, y



def create_sets(new_data_w_index, encoded_labels, train_index, test_index, max_len, txt_flag):

    ''' 
    Purpose: This function creates the Train, Validation and Test sets. We keep a ratio of Train:Validation:Test as 64:16:20
    Input:  Input data (new_data_w_index): Contains text inputs and indices for images.(input_ids, attention_mask, token_ids) from Transformer Encoder.
            Encoded Labels: Encoded Lables
            Train Index: Indixes of samples in the training set obtained using 5-fold cross validation.
            Test Index: Indixes of samples in the testing set obtained using 5-fold cross validation.
            max_len: Maximum length of the textual input.
            txt_flag: Indicating whether the tranformer model used is BERT or RoBERTa. RoBERTa doesn't require token_ids and hence we use this flag.
    Output: training_set, validation_set, test_set
            test_data, metric_test: test_data is the data which will be used for testing. metric_test contains test labels.
    '''

	train_data, test_data = new_data_w_index[train_index], new_data_w_index[test_index]
	train_label, test_label = encoded_labels[train_index], encoded_labels[test_index]

	train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, stratify=train_label, test_size=0.2, random_state=42)

	train_label = to_categorical(train_label)
	validation_label = to_categorical(validation_label)

	metric_test = np.copy(test_label)
	test_label = to_categorical(test_label)

	train_img_indices = np.stack(train_data[:,0])
	print(train_img_indices)
	train_data = np.stack(train_data[:,1])

	validation_img_indices = np.stack(validation_data[:,0])
	validation_data = np.stack(validation_data[:,1])

	test_img_indices = np.stack(test_data[:,0])
	test_data = np.stack(test_data[:,1])

	train_text_input_ids = np.copy(train_data[:,0:max_len])
	validation_text_input_ids = np.copy(validation_data[:,0:max_len])
	test_text_input_ids = np.copy(test_data[:,0:max_len])

	train_text_attention_mask = np.copy(train_data[:,max_len:2*max_len])
	validation_text_attention_mask = np.copy(validation_data[:,max_len:2*max_len])
	test_text_attention_mask = np.copy(test_data[:,max_len:2*max_len])

	if txt_flag == 0:
	    train_text_token_ids = np.copy(train_data[:,2*max_len:3*max_len])
	    validation_text_token_ids = np.copy(validation_data[:,2*max_len:3*max_len])
	    test_text_token_ids = np.copy(test_data[:,2*max_len:3*max_len])
	else:
	    train_text_token_ids = None
	    validation_text_token_ids = None
	    test_text_token_ids = None

	print("Creating Datasets...")
	#         print(train_text_input_ids.shape, train_text_attention_mask.shape, train_text_token_ids.shape)
	training_set = Dataset(train_text_input_ids, train_text_attention_mask, train_text_token_ids, train_img_indices, train_label, "train")
	test_set = Dataset(test_text_input_ids, test_text_attention_mask, test_text_token_ids, test_img_indices, test_label, "test")
	validation_set = Dataset(validation_text_input_ids, validation_text_attention_mask, validation_text_token_ids, validation_img_indices, validation_label, "validation")


	return training_set, validation_set, test_set, test_data, metric_test