#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mini VISICS specific script to plot the caltech evaluation results on the INRIA dataset
"""

from __future__ import print_function

#import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")
 

import os, os.path, shutil, subprocess, glob
from optparse import OptionParser

from detections_to_caltech import detections_to_caltech

#at_visics = True
at_visics = False

def plot_caltech_evaluation(results_path, disable_show, exp_idx ):

    results_path = os.path.normpath(results_path)
    results_name = os.path.split(results_path)[-1]

    data_sequence_path = os.path.join(results_path, "detections.data_sequence")
    v000_path  = os.path.join(results_path, "V000")

    if os.path.exists(v000_path):
        print(v000_path, "already exists, skipping the generation step")
    else:
        # convert data sequence to caltech data format
        detections_to_caltech(data_sequence_path, v000_path)

    if at_visics:
        caltech_evaluation_dir = "/esat/enif/bgunyel/DataSets/CalTechEvaluation"
    else:            
         caltech_evaluation_dir = "/home/rodrigob/data/CalTechEvaluation_data"
    caltech_results_dir = os.path.join(caltech_evaluation_dir, "data-INRIA/res/Ours-wip")
    
    set01_path = os.path.join(caltech_results_dir, "set01_" + results_name)
    set01_v000_path = os.path.join(set01_path, "V000")
    if os.path.exists(set01_v000_path):
        print(set01_v000_path, "already exists, skipping the copy step")
    else:
        os.mkdir(set01_path)    
        shutil.copytree(v000_path, set01_v000_path)    

    # update the set01 symbolic link    
    set01_ln_path = os.path.join(caltech_results_dir, "set01")
    os.remove(set01_ln_path)
    os.symlink(set01_path, set01_ln_path)

    # clear previous evaluations    
    caltech_eval_dir = os.path.join(caltech_evaluation_dir, "data-INRIA/eval")    
    paths_to_remove = glob.glob(os.path.join(caltech_eval_dir, "*"))
    for path in paths_to_remove:
        os.remove(path)

    # run the matlab code
    current_dir = os.getcwd()
    os.chdir(caltech_evaluation_dir)
    if at_visics:
        matlab_command = 'matlab -nodisplay -r "dbEval ('+str(exp_idx)+'); quit"'
    else:
        #matlab_command = 'octave --eval "dbEvalOctave; quit"'
        matlab_command = '/home/rodrigob/Downloads/matlab_install/bin/matlab -nodisplay -r "dbEval; quit"'


    print("Running:", matlab_command)
    subprocess.call(matlab_command, shell=True)
    os.chdir(current_dir)
    
    if at_visics:
        # copy and rename the the curve of our detector    
        result_mat_path = os.path.join(set01_path, results_name + "Ours-wip.mat")    
        shutil.copy(os.path.join(caltech_eval_dir, "Ours-wip.mat"), result_mat_path)

    # copy and rename the resulting pdf    
    result_pdf_path = os.path.join(set01_path, results_name + "_roc.pdf")    
    pdfname = "INRIA roc exp=" + getExperimentName(exp_idx) +".pdf"
    shutil.copy(os.path.join(caltech_eval_dir, pdfname), result_pdf_path)

    
    # done
    print("Done creating the evaluation. Check out %s." % result_pdf_path)
    if not disable_show:
		subprocess.call("gnome-open %s" % result_pdf_path, shell=True)    
    return

def getExperimentName(index):
    exps = {}

    exps[1] ='all'
    exps[2] ='reasonable'
    exps[3] ='scale=near'
    exps[4] ='scale=medium'
    exps[5] ='scale=far'
    exps[6] ='occ=none'
    exps[7] ='occ=partial'
    exps[8] ='occ=heavy'
    exps[9] ='ar=all'
    exps[10] ='ar=typical'
    exps[11] ='ar=atypical'
    exps[12] ='octave=-1'
    exps[13] ='octave=0'
    exps[14] ='octave=1'
    exps[15] ='octave=2'
    exps[16] ='octave=3'
    return exps[index]

def main():

    parser = OptionParser()
    parser.description = \
        """
        Reads the recordings of objects_detection over Caltech version of INRIA dataset, 
        calls the Caltech evaluation toolbox, and shows the resulting plot
        """

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the recording directory")
    parser.add_option("-d", "--disable_show", dest="disable_show",
                        action="store_true",default=False,
                       help="disable showing of the resulting pdf file")
    parser.add_option("-e", "--exp_idx", dest="exp_idx",
                        type="int",default=1,
                       help="specify which experiment setup to use")

    (options, args) = parser.parse_args()
    #print (options, args)
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input directory")
    else:
        parser.error("'input' option is required to run this program")

    if not os.path.isdir(options.input_path):
        parser.error("the 'input' option should point towards " \
                     "the recording directory of the objects_detection application")

    results_path = options.input_path

    print ("show result disabled=" , options.disable_show)
    plot_caltech_evaluation(results_path, options.disable_show, options.exp_idx)

    return
        

if __name__ == '__main__':
    main()
    
    
