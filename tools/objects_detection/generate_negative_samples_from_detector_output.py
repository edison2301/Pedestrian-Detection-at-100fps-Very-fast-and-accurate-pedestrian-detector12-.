#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Markus: please add description here
"""

from __future__ import print_function
import Image
import glob
import sys
import random
import os
from create_multiscales_training_dataset import create_positives, Box
from detections_to_precision_recall import open_data_sequence


from optparse import OptionParser

def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates occluded pedestrians"

    parser.add_option("-i", "--input", dest="input_file",
                       metavar="PATH", type="string",
					    help="path to the .data_sequence file")


    parser.add_option("-o", "--output", dest="output_path",
                       metavar="DIRECTORY", type="string",
                       help="path to a non existing directory where the new training dataset will be created")
                                                  
    parser.add_option("-n", "--number_of_samples", dest="number_of_samples",
                       type="int",default=10,
                       help="Number of samples to be randomly choosen")
                                                  
    (options, args) = parser.parse_args()

    if options.input_file:
		pass
    else:
        parser.error("'input_file' option is required to run this program")

    #if not options.output_path:
    #    parser.error("'output' option is required to run this program")
    #    if os.path.exists(options.output_path):
	#		parser.error("output_path already exists")

    return options 


def get_annotations(detections_sequence, number):
	output_detections = []
	negCount = 0
	posCount = 0
	all_detections= []

	for detections in detections_sequence:
		detections.image_name
		for detection in detections.detections:
			box = Box()
			box.min_corner.x = detection.bounding_box.min_corner.x
			box.min_corner.y = detection.bounding_box.min_corner.y
			box.max_corner.x = detection.bounding_box.max_corner.x
			box.max_corner.y = detection.bounding_box.max_corner.y
			
			detection_data = [os.path.join('/esat/kochab/mmathias/INRIAPerson/Train/neg/', detections.image_name), detection.score, box]
			all_detections.append(detection_data)
	
			
		
#sort detections by score
	all_detections= sorted(all_detections, key=lambda det: det[1],reverse=True)
	#print (all_detections[:100])
	annotations = []
	if len(all_detections) > number:
		all_detections = all_detections[:number]
	else:
		print("number of detections provided: ", len(all_detections))
		raise Exception("not enough detections provided")
	
	
	#sort detections by filename
	all_detections= sorted(all_detections, key=lambda det: det[0],reverse=True)
	#print (all_detections[:100])
	previous_filename = ""
	boxes = []
	for det in all_detections:
		fn = det[0]
		
		if fn == previous_filename:
			boxes.append(det[2])

		else:
			if (previous_filename != ""):
				annotation =(previous_filename, boxes)
				annotations.append(annotation)
			previous_filename = det[0]
			boxes = [det[2],]
	#print(annotations)
	return (annotations)




	return [output_detections, posCount, negCount]


def main():
	options = parse_arguments()	
	os.mkdir(options.output_path)
	number_of_samples = options.number_of_samples 
	

#get annotations
	detections_sequence = open_data_sequence(options.input_file)

	annotations = get_annotations(detections_sequence, number_of_samples)

#generate data
	model_width, model_height = 64, 128
	cropping_border = 20 # how many pixels around the model box to define a test image ?
	octave=0

	create_positives(model_width, model_height, cropping_border, octave, annotations, options.output_path)
	


if __name__ == "__main__":
	main()

