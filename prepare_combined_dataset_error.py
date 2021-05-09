import os, sys
from transformers import BertConfig, BertTokenizer, RobertaTokenizer, RobertaConfig
import numpy as np
from twitter_text_preprocessing import *
from twitter_testing import *


def make_bert_input_w_OCR(text_data, ocr_data, max_len, tokenizer):
	''' 
    Purpose: This function encodes the text along with OCR using the BERT encoder.
    Input:  text_data: Text data
            ocr_data: OCR Text data
            max_len : Maximum length of the sequence
			tokenizer: BERT tokenizer from Huggingface
    Output: new_data: [input_ids,attention_masks,token_ids]
	'''
	input_ids = []
	attention_masks = []
	token_ids = []
	for i in range(len(text_data)):
		encoded_dict = tokenizer.encode_plus(
							text_data[i],                      # Sentence to encode.
							text_pair = ocr_data[i],
							add_special_tokens = True, # Add '[CLS]' and '[SEP]'
							max_length = max_len,           # Pad & truncate all sentences.
							pad_to_max_length = True,
							return_attention_mask = True,   # Construct attn. masks.
							return_token_type_ids = True,
					   )

		# Add the encoded sentence to the list.    
		input_ids.append(encoded_dict['input_ids'])

		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

		token_ids.append(encoded_dict['token_type_ids'])

	input_ids = np.asarray(input_ids, dtype='int32')
	attention_masks = np.asarray(attention_masks, dtype='int32')
	token_ids = np.asarray(token_ids, dtype='int32')

	new_data = np.concatenate((input_ids, attention_masks), axis = 1)
	new_data = np.concatenate((new_data, token_ids), axis=1)

	return new_data

def make_roberta_input_w_OCR(text_data, ocr_data, max_len, tokenizer):
	''' 
    Purpose: This function encodes the text along with OCR using the RoBERTa encoder.
    Input:  text_data: Text data
            ocr_data: OCR Text data
            max_len : Maximum length of the sequence
			tokenizer: RoBERTa tokenizer from Huggingface
    Output: new_data: [input_ids,attention_masks]
	'''
	input_ids = []
	attention_masks = []
	token_ids = []
	for i in range(len(text_data)):
		encoded_dict = tokenizer.encode_plus(
							text_data[i],                      # Sentence to encode.
							text_pair = ocr_data[i],
							add_special_tokens = True, # Add '[CLS]' and '[SEP]'
							max_length = max_len,           # Pad & truncate all sentences.
							pad_to_max_length = True,
							return_attention_mask = True,   # Construct attn. masks.
							add_prefix_space = True,
					   )

		# Add the encoded sentence to the list.    
		input_ids.append(encoded_dict['input_ids'])

		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

		#token_ids.append(encoded_dict['token_type_ids'])

	input_ids = np.asarray(input_ids, dtype='int32')
	attention_masks = np.asarray(attention_masks, dtype='int32')
	#token_ids = np.asarray(token_ids, dtype='int32')

	new_data = np.concatenate((input_ids, attention_masks), axis = 1)
	#new_data = np.concatenate((new_data, token_ids), axis=1)

	return new_data


def get_dataset(datafile, data_col, ocr_col, label_col, txt_flag, multilabel, ocr_flag):

	''' 
    Purpose: This function gets the dataset file name and other optional flags to encoded data along with the labels
    Input:  data_file: path of the dataset file.
            data_col: column number for the data (text data).
			ocr_col: column number for the OCR data.
            label_col : column number for the labels
			txt_flag: Flag indicating whether to use BERT (0) or RoBERTa (1)
			multilabel: Flag indicating whether the setting is for binary labels (0) or multiclass labels (1)
			ocr_flag: Flag indicating whether to use OCR or not.
    Output: new_data_w_index: Contains the new encoded data with indices for getting the images. 
			encoded_labels: Encoded labels
			text_data: Text data obtained from the source datasite file after pre-processing
			ocr_data: OCR data obtained from the source datasite file after pre-processing
	'''

	text_data, labels = prepare_dataset(datafile, data_col, label_col, "word-based")
	ocr_data, labels = prepare_dataset(datafile, ocr_col, label_col, "word-based")
	print(len(text_data))

	max_len = 100
	if txt_flag == 0:
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
		if ocr_flag == 1:
			new_data = make_bert_input_w_OCR(text_data, ocr_data, max_len, tokenizer)
		else:
			new_data = make_bert_input(text_data, max_len)
	else:
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
		if ocr_flag == 1:
			new_data = make_roberta_input_w_OCR(text_data, ocr_data, max_len, tokenizer)
		else:
			new_data = make_roberta_input(text_data, max_len)

	new_data_w_index = []

	multi_text_data = []
	multi_ocr_data = []
			
	if multilabel == 1:
		encoded_labels = []
		encoding = {
			'0' : 0,
			'1' : 1,
			'2' : 2,
			'3' : 3
		}
		for i in range(len(new_data)):
			if labels[i] in ['0', '1', '2', '3']:
				new_data_w_index.append(np.array([[str(i+1)], new_data[i]]))
				encoded_labels.append(encoding[str(int(labels[i]))])
				multi_text_data.append(text_data[i])
				multi_ocr_data.append(ocr_data[i])

		text_data = multi_text_data
		ocr_data = multi_ocr_data

		print(len(multi_text_data))

	else:
		encoded_labels = []
		encoding = {
			'0' : 0,
			'1' : 1
		}
		for i in range(len(new_data)):
			new_data_w_index.append(np.array([[str(i+1)], new_data[i]]))
			encoded_labels.append(encoding[str(int(labels[i]))])


	new_data_w_index = np.array(new_data_w_index)
	encoded_labels = np.array(encoded_labels)


	return new_data_w_index, encoded_labels, text_data, ocr_data