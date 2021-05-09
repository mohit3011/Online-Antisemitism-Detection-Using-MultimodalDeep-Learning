import sys
import torch
torch.manual_seed(42)
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os,sys,re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, RobertaConfig
import statistics
import warnings
import pickle
warnings.filterwarnings("ignore")
from create_dataloader_sets import *
from training_loop_encoder import *
from experiment_check import *
from prepare_combined_dataset_error import get_dataset
import csv


class multiModel(nn.Module):
  
    def __init__(self, num_labels=2, config=None, txt_flag=0, img_flag=2, device=torch.device("cuda:0")):
        ''' 
        Purpose: Class for creating the proposed multimodal architecture
        Input:  num_labels: Numbers of labels (Binary==2, Multiclass==4)
                config: Config file for the transformer models (if available)
                txt_flag : Flag for determining the Tranformer model to be used. 0: BERT, 1: RoBERTa
                img_flag: Flag for detemining the Image model to be used. 0: RESNET, 2: DenseNet
        '''
        
        super(multiModel, self).__init__()
        
        # Common layers
        self.bn = nn.BatchNorm1d(384, momentum=0.99)
        self.dense1 = nn.Linear(in_features=384, out_features=128) #Add ReLu in forward loop
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features=128, out_features=num_labels)
        self.device = device
        self.imagegate = nn.Sigmoid()

        self.encoder_dense_1 = nn.Linear(in_features=2816, out_features=768)
        self.encoder_dense_2 = nn.Linear(in_features=768, out_features=384)

        self.decoder_dense_1 = nn.Linear(in_features=384, out_features=768)
        self.decoder_dense_2 = nn.Linear(in_features=768, out_features=2816)
        

        self.s_text_vector = []
        self.s_image_vector = []
        self.h_text_vector = []
        self.h_image_vector = []
        self.imagegate = nn.Sigmoid()
        self.intermediate_layer_output = {'output': None}

        self.s_text_vector = np.asarray(np.random.choice([-1,1], 768), dtype='int32')
        self.s_image_vector = np.asarray(np.random.choice([-1,1], 768), dtype='int32')

        # Txt model
        if txt_flag == 0:
            self.txt_model = BertModel.from_pretrained('bert-base-uncased', config=config)
        else:
            self.txt_model = RobertaModel.from_pretrained('roberta-base', config=config)
        
        # image model
        if img_flag == 0:
            self.img_model = torchvision.models.resnet152(pretrained=True)
            self.img_model.layer2[-1].register_forward_hook(self.setIntermediateValue) #Adding a forward hook for getting intermediate output
            self.intermediate_img_model = nn.AvgPool2d(kernel_size=28, stride=1, padding=0)
            num_ftrs = self.img_model.fc.in_features
            self.img_model.fc = nn.Linear(num_ftrs, 768)
        else:
            self.img_model = torchvision.models.densenet161(pretrained=True)
            self.img_model.features.denseblock4.register_forward_hook(self.setIntermediateValue) #Adding a forward hook for getting intermediate output
            
            self.intermediate_img_model = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            num_ftrs = self.img_model.classifier.in_features
            self.img_model.classifier = nn.Linear(num_ftrs, 768)

    def setIntermediateValue(self, module, input, output):
        self.intermediate_layer_output['output'] = output

    def encoder(self, combined_features):

        encoded_vector_dense_1 = self.encoder_dense_1(combined_features)
        encoded_feature_vector = self.encoder_dense_2(encoded_vector_dense_1)

        return encoded_feature_vector

    def decoder(self, encoded_vector):

        decoded_vector_dense_1 = self.decoder_dense_1(encoded_vector)
        decoded_features = self.decoder_dense_2(decoded_vector_dense_1)

        return decoded_features

    def forward(self, inputs, imgs, attention_mask=None, labels=None):

        text_input_ids_in = inputs[:,0,:].long()
        text_input_masks_in = inputs[:,1,:].long()
        
        if txt_flag == 0:
            text_input_token_in = inputs[:,2,:].long().to(self.device)
            text_embedding_layer = self.txt_model(text_input_ids_in, attention_mask=text_input_masks_in, token_type_ids=text_input_token_in)[0]
        else:
            text_embedding_layer = self.txt_model(text_input_ids_in, attention_mask=text_input_masks_in)[0] #RoBERTa doesn't require token_ids
        
        text_cls_token = text_embedding_layer[:,0,:]
        #print("text_cls_token.shape: ", text_cls_token.shape)
        
        img_features = self.img_model(imgs)
        intermediate_pooled_img_features = self.intermediate_img_model(self.intermediate_layer_output['output'])
        intermediate_img_features = torch.flatten(intermediate_pooled_img_features, start_dim=1, end_dim=-1).to(self.device)
        #print("intermediate_features.shape :", intermediate_features.shape)

        intermediate_features = self.imagegate(torch.cat((text_cls_token, intermediate_img_features), 1))
        #print(intermediate_features.shape)
        final_features = self.imagegate(torch.cat((torch.cat((text_cls_token, intermediate_features), 1), img_features), 1))
       

        encoded_vector = self.encoder(final_features)

        decoded_features = self.decoder(encoded_vector)
        
        X = self.bn(encoded_vector)
        X = F.relu(self.dense1(X))
        X = self.dropout(X)
        X = F.log_softmax(self.dense2(X))
        
        return X, final_features, decoded_features


if __name__ == '__main__':

    ''' Arguments to be given by the user
        multilabel: Whether the classification is Binary=0, Multiclass=1
        img_flag: For using the pre-trained Image models. 0 for ResNet152 2 for DenseNet161
        txt_flag: For using the Transformer models. 0 for BERT, 1 for RoBERTa
        alpha, beta and custom_loss: For customising the loss function (not used here)
    '''
    multilabel = int(sys.argv[1])
    img_flag = int(sys.argv[2])
    txt_flag = int(sys.argv[3])
    alpha = float(sys.argv[4])
    beta = float(sys.argv[5])
    custom_loss = int(sys.argv[6])

    
    ocr_flag = 1 # 0 for model with OCR, 1 for without OCR

    datafile = "combined_dataset.csv"
    data_col = 2
    ocr_col = 3
    
        
    max_len = 100 #Maximum length of text and OCR combined
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    '''
    For calculating the metrics
    '''
    acc_cum = 0
    rec_cum = 0
    pre_cum = 0
    f1_cum = 0
    f1_micro_cum = 0
    acc_arr = []
    rec_arr = []
    pre_arr = []
    f1_arr = []
    f1_micro_arr = []
    predicted_label_arr = []
    test_label_arr = []
    error_analysis = []

    #######################################################################################################


    print_configuration(img_flag, multilabel, txt_flag, ocr_flag)

    new_data_w_index, encoded_labels, text_data, ocr_data = get_dataset(datafile, data_col, ocr_col, label_col, txt_flag, multilabel, ocr_flag)

    # In[9]:

    print("Number of Examples:, ", new_data_w_index.shape[0])


    class_weights_labels = class_weight.compute_class_weight('balanced',
                                                 np.unique(encoded_labels),
                                                 encoded_labels)
    print("Number of Classes: ", len(class_weights_labels))


    if txt_flag == 0:
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=len(class_weights_labels))
    else:
        config = RobertaConfig.from_pretrained('roberta-base', num_labels=len(class_weights_labels))

    config.output_hidden_states = False


    # In[13]:


    encoded_labels = np.asarray(encoded_labels, dtype='int32')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights_labels = torch.tensor(class_weights_labels, dtype=torch.float, device=device)


    # In[ ]:


    fold_number = 1
    for train_index, test_index in skf.split(new_data_w_index, encoded_labels):
            print("Running fold #", fold_number)
            
            training_set, validation_set, test_set, test_data, metric_test = create_sets(new_data_w_index, encoded_labels, train_index, test_index, max_len, txt_flag)
            
            dataloaders = {
                'train' : torch.utils.data.DataLoader(training_set, batch_size=4,
                                                     shuffle=True, num_workers=2, drop_last=True),
                'test' : torch.utils.data.DataLoader(test_set, batch_size=4,
                                                     shuffle=True, num_workers=2, drop_last=True),
                'validation': torch.utils.data.DataLoader(validation_set, batch_size=4,
                                                     shuffle=True, num_workers=2, drop_last=True)
            }

            dataset_sizes = {
                'train': len(training_set),
                'test' : len(test_set),
                'validation' : len(validation_set)
            }
            
            print("Training network...")
            model = multiModel(num_labels=len(class_weights_labels), config=config, txt_flag=txt_flag, img_flag=img_flag)
            model = train_loop_encoder(model, dataloaders, dataset_sizes, class_weights_labels, alpha, beta, custom_loss, num_epochs=99)
    #         save_models(1, model)
            test_img_indices = np.stack(new_data_w_index[test_index][:,0])
            
            y_pred = np.array([])
            model.eval()
            for i in range(len(test_set)):
                    inputs, imgs, _ = test_set[i]
                    inputs = torch.Tensor(np.expand_dims(inputs, axis=0)).to(device)
                    imgs = torch.Tensor(np.expand_dims(imgs, axis=0)).to(device)
                    outputs, combined_features, decoded_features = model(inputs, imgs)
                    preds = torch.max(outputs, 1)[1]
                    y_pred = np.append(y_pred, preds.cpu().numpy())
                    
            test_label_arr.extend(metric_test)
            predicted_label_arr.extend(y_pred)

            acc_arr.append(accuracy_score(metric_test, y_pred))
            acc_cum += acc_arr[fold_number-1]
            rec_arr.append(recall_score(metric_test, y_pred, average='macro'))
            rec_cum += rec_arr[fold_number-1]
            pre_arr.append(precision_score(metric_test, y_pred, average='macro'))
            pre_cum += pre_arr[fold_number-1]
            f1_arr.append(f1_score(metric_test, y_pred, average='macro'))
            f1_cum  += f1_arr[fold_number-1]
            f1_micro_arr.append(f1_score(metric_test, y_pred, average='micro'))
            f1_micro_cum  += f1_micro_arr[fold_number-1]
            
                
            test_text_data = np.array(text_data)[test_index]
            test_ocr_data = np.array(ocr_data)[test_index]

            for i in range(len(test_text_data)):
                if int(metric_test[i])!=int(y_pred[i]):
                    temp_row = []
                    temp_row.append(test_text_data[i])
                    temp_row.append(test_ocr_data[i])
                    temp_row.append(test_img_indices[i][0])
                    temp_row.append(metric_test[i])
                    temp_row.append(y_pred[i])
                    error_analysis.append(temp_row)
            
            fold_number+=1



    print("Accuracy: ", acc_cum/5)
    print("Recall: ", rec_cum/5)
    print("Precision: ", pre_cum/5)
    print("F1 score(macro): ", f1_cum/5)
    print("F1 score(micro): ", f1_micro_cum/5)

    print("Accuracy_stdev: ", statistics.stdev(acc_arr))
    print("Recall_stdev: ", statistics.stdev(rec_arr))
    print("Precision_stdev: ", statistics.stdev(pre_arr))
    print("F1(macro) score_stdev: ", statistics.stdev(f1_arr))
    print("F1(micro) score_stdev: ", statistics.stdev(f1_micro_arr))

    confusion_matrix_final = confusion_matrix(test_label_arr, predicted_label_arr)
    print(confusion_matrix_final)
        
    print_configuration(img_flag, multilabel, txt_flag, ocr_flag)