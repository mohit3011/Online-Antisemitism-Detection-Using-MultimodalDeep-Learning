import os, sys


def print_configuration(img_flag, multilabel, txt_flag, ocr_flag):
	
	if img_flag == 0:
		im_net = "resnet152"
	elif img_flag == 1:
		im_net = "resnext"
	else:
		im_net = "densenet161"

	if multilabel == 1:
		mul = "(multilabel)"
	else:
		mul = "(binary)"

	if txt_flag == 0:
		tx_net = "BERT"
	else:
		tx_net = "RoBERTa"

	if ocr_flag == 1:
		modality = "Image_OCR_Text"
	else:
		modality = "Image_Text"

	print("Configuration is", modality, im_net, tx_net, mul)