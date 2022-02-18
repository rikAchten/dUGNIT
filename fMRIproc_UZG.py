#!/usr/bin/env python
# coding: utf-8

# # Definitions and basics
# Load necessary items

#get_ipython().run_line_magic('pylab', 'inline')

from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True) # Set numpy to print only 2 decimal digits for neatness

import warnings
warnings.filterwarnings('ignore')

import os, os.path
from os.path import join as opj
import json, pprint
import shutil, time
import nibabel as nib

import re
import unicodedata

import csv
import operator

import pydicom

from datetime import datetime
from time import strftime

from pathlib import Path

import platform

from nilearn import image as nli
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_epi
from nilearn.plotting import plot_anat
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import plot_stat_map

from IPython.display import SVG, Markdown, display, HTML, Image, clear_output
START = '\033[4m' # start underline
END = '\033[0m' # stop underline

import pandas as pd

from nipype.interfaces.dcm2nii import Dcm2niix

from builtins import str
from builtins import range

from nipype import config
from nipype import SelectFiles

from nipype.algorithms.misc import Gunzip

from nipype.interfaces import spm, fsl

from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/opt/spm12')

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
import nipype.algorithms.confounds as cnfds # for TSNR
import nipype.interfaces.matlab as matlb

from nipype.interfaces.base import Bunch
from nipype.algorithms.modelgen import SpecifySPMModel


# ## Some general variables

# In[3]:


wDir = '/data/Patients'
oDir = '/data/output'
iDir = '/data/Patients/SubjectInfoFiles'
patDcmRoot = '/data/Patients/tmp_dicom'
bidsSource = '/data/BIDS_source'

# folders and files in the subject directory
folders = ['anat', 'asl', 'func', 'dwi', 'fmap', 'not_used'] # deleted 'pwi'

# dictionary linking SeriesDescription to folders
fDict = {'flair3d': 'anat' , 'mprage':'anat', 'swi':'anat', 't2star':'anat',
          'wgen':'func', 'read':'func', 'categ':'func', 'enco_old':'func', 'enco_new':'func', 'encoold':'func', 'enconew': 'func',
          'motoriek_handen':'func', 'motoriek_voeten':'func', 'motoriek_mond':'func', 
          'visus':'func',
          'sensibiliteit_handen':'func',
          'restingstate':'func',
          'dti':'dwi', 'diff': 'dwi', 'field_map':'fmap', 'asl':'asl', 'pwi': 'pwi'}

# dictionary defining the number of imags in a functional run, if lower, the run is not complete, double values for SMS (PRISMA)
fimgDict = {
             'WGEN': [120, 240], 'READ':[120, 240], 'CATEG':[120,240], 'ENCO_OLD':[96, 192], 'ENCO_NEW':[360,720], 'ENCOOLD':[96, 192], 'ENCONEW':[360,720],
             'MOTORIEK_HANDEN':[120, 160, 240], 'MOTORIEK_VOETEN':[120, 160, 240], 'MOTORIEK_MOND':[120, 160, 240],
             'SENSIBILITEIT_HANDEN':[120, 160, 240], 
             'VISUS':[120, 160, 240],
             'RESTINGSTATE': [360],
             'field_map':[1], 'T1_mprage': [1]
            }

paradigmList = [x for x in fDict.keys() if fDict[x] == 'func']

# Some hyperparameters
# Define an empty list here
slicetimingParam = []
# Define the smoothing kernel FWHM for susan
susanFWHM = 6
# Specify the isometric voxel resolution you want after coregistration
desiredVoxelIso = 3

bbrPath ='/usr/local/fsl/etc/flirtsch/bbr.sch'

# Use the following tissue specification to get a GM and WM probability map
# This is installation specific (spm12 on Darwin <homedir>/spm12)

if platform.system() == 'Linux':
    tpmImg ='/opt/spm12/tpm/TPM.nii'
elif platform.system() == 'Darwin':
    tpmImg = homedir + '/spm12/tpm/TPM.nii'
#elif platform.system() == 'Windows':
#   tpm_img ='/opt/spm12/tpm/TPM.nii

tissue1 = ((tpmImg, 1), 1, (True,False), (False, False))
tissue2 = ((tpmImg, 2), 1, (True,False), (False, False))
tissue3 = ((tpmImg, 3), 2, (True,False), (False, False))
tissue4 = ((tpmImg, 4), 3, (False,False), (False, False))
tissue5 = ((tpmImg, 5), 4, (False,False), (False, False))
tissue6 = ((tpmImg, 6), 2, (False,False), (False, False))
tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]

# Variables defined during processing
# dcmdir: the name of the dicm dir in /data/Patients/tmp_dicom
# pat2Proc: a dictionary containing dicom info of the patient to process
# patID: a concatenation of sub-<Accession Number, derived from pat2Proc>

# onsets for enconew, in scans
onsetNew = [4, 5, 6, 8, 9, 11, 19, 21, 22, 23, 28, 34, 35, 36, 37, 45, 46, 48, 49, 53, 60, 61, 62, 67, 
        73, 74, 75, 76, 79, 81, 88, 89, 91, 92, 93, 95, 99, 105, 106, 107, 110, 111, 124, 126, 127, 
        128, 130, 131, 140, 141, 142, 143, 144, 150, 151, 160, 161, 162, 164, 166, 174, 176, 177, 
        178, 182, 183, 184, 185, 186, 192, 193, 200, 201, 202, 203, 205, 213, 214, 216, 217, 219, 
        272, 274, 281, 282, 284, 285, 286, 292, 298, 300, 301, 302, 309, 310, 311, 316, 322, 323, 
        324, 325, 328, 330, 337, 340, 341, 342, 344, 348, 354, 355, 356]

onsetOld = [1, 2, 3, 7, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 25, 26, 27, 29, 30, 31, 32, 33, 38, 39, 
        40, 41, 42, 43, 44, 47, 50, 51, 52, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 68, 69, 70, 71, 
        72, 77, 78, 80, 82, 83, 84, 85, 86, 87, 90, 94, 96, 97, 98, 100, 101, 102, 103, 104, 108, 109, 
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 129, 132, 133, 134, 135, 136, 
        137, 138, 139, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159, 163, 165, 167, 
        168, 169, 170, 171, 172, 173, 175, 179, 180, 181, 187, 188, 189, 190, 191, 194, 195, 196, 197, 
        198, 199, 204, 206, 207, 208, 209, 210, 211, 212, 215, 218, 221, 222, 223, 224, 225, 226, 227, 
        228, 229, 231, 233, 237, 238, 239, 240, 243, 246, 247, 248, 249, 250, 251, 252, 256, 257, 258, 
        259, 261, 262, 263, 264, 265, 270, 271, 273, 275, 276, 277, 278, 279, 280, 283, 287, 288, 289, 
        290, 291, 293, 294, 295, 296, 297, 299, 303, 304, 305, 306, 307, 308, 312, 313, 314, 315, 317, 
        318, 319, 320, 321, 326, 327, 329, 331, 332, 333, 334, 335, 336, 338, 339, 343, 345, 346, 347, 
        349, 350, 351, 352, 353, 357, 358, 359, 360]

highPassFilterCutoffDict = {'wgen': 128, 'read': 128, 'categ': 128,'enco_old': 96, 'enco_new': 120, 'encoold': 96, 'enconew': 120, 'restingstate': 120}


# ## Functions

# ### Trivia

# In[4]:


def stripaccents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


# In[5]:


def text2id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = stripaccents(text.lower())
    text = re.sub('[ ]+', '_', text)
    text = re.sub('[^0-9a-zA-Z_-]', '', text)
    return text


# In[6]:


def makePatList():
    patList = os.listdir(patDcmRoot)
    #Get rid of accented characters and white space ..., everything small caps
    for i in range(len(patList)):
        if patList[i] != text2id(patList[i]):
            os.rename(opj(patDcmRoot, patList[i]), opj(patDcmRoot, text2id(patList[i])))
    patList = os.listdir(patDcmRoot)
    return patList


# In[7]:


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")


# In[8]:


# Select WM segmentation file from [segmentation output]
def get_wm(files):
    return files[1][0]

# Select GM segmentation file from [segmentation output]
def get_gm(files):
    return files[0][0]


# ### General BIDS & DICOM

# In[9]:


def create_bids_subfolders(folders):
    """
    Function to create the fMRI etc sub-folders in the current dir
    folders is a defined general variable
    
    """

    for folder in folders:
        os.mkdir(folder)


# In[10]:


def mkbidsdir(pat2Proc):
    """
    Function to check and create the subjects bids dir in wDir and sub-dirs in the subDir
    subDir is the directory to create in wDir
    
    """
    
    subID = 'sub-' + pat2Proc['Accession Number']
    subDir = opj(wDir, subID)
    
    os.chdir(wDir)
    # Make de subDir if it doesn't exist
    if not os.path.exists(subDir):
        print("Creating '" + str(subDir) + "'.")
        os.mkdir(subDir)
    
    # Make sure the subDir is empty, then create subdirs
    if len([name for name in os.listdir(subDir)]) != 0:
        print("'" + str(subDir) + "' is not empty.")
        os.listdir(subDir)
        print("Nothing changed, please check!")
    else:
        print("Creating subdirectories in '" + str(subDir) + "'.")
        os.chdir(subDir)
        create_bids_subfolders(folders)
        os.chdir('..')


# In[11]:


def get_dicom_info(subDcmDir):
    """
    Function to acquire dicom information of the subject/patient
    subDcmDir = the subjects dicom directory in /data/Patients/tmp_dicom 
    
    returns: dcmtagsDict = {'Patient ID': PatientID, 'Date of Birth':DOB, 'Accession Number': AccessionNumber, 'Study Date': StudyDate}
    
    """

    dcmfileList = []
    dcmtagsDict = {}

    os.chdir(opj(patDcmRoot, subDcmDir))
    
    for ff in os.listdir('.'):
        if ff.lower().endswith('dcm') or ff.lower().endswith('ima'):
            dcmfileList.append(ff)
        
    dcmfileList = sorted(dcmfileList)
    
    if len(dcmfileList) == 0:
        print('No dicom files in %s.' % subDcmDir)
        subPatName = 'Unknown'
        subAccNum = 'Unknown'
        subStuDate = 'Unknown'
        subDOB = 'Unknown'
        subPatID = 'Unknown'
        
    else:
        dcmfile = pydicom.read_file(dcmfileList[0], force=True)
        subPatName = dcmfile.PatientName
        subPatName = str(subPatName).replace('^', ' ')
        subAccNum = str(dcmfile.AccessionNumber)
        subStuDate = str(dcmfile.StudyDate)
        subDOB = str(dcmfile[0x0010, 0x0030].value)
        subPatID = str(dcmfile.PatientID)
    
    dcmtagsDict['Patient Name'] = subPatName
    dcmtagsDict['Patient ID'] = subPatID
    dcmtagsDict['Date of Birth'] = subDOB
    dcmtagsDict['Accession Number'] = subAccNum
    dcmtagsDict['Study Date'] = subStuDate
    
    return dcmtagsDict


# ### Select Patient Function

# In[12]:


def select_patient_fMRI(sortcriterion = 'name'):
    """
    Function to select a patient not yet processed in /data/Patients/tmp_dicom
    Reads dicom tags from first file in /data/Patients/tmp_dicom dirs
    Compares accession number to subjects in /data/Patients
    
    sortcriterion can be 'name' (default) or 'date'
    
    Output: notProcessedList, same but only with patients not processed in /data/Patients:
            {'Patient ID': *, 'Date of Birth': *, 'Accession Number': *, 'Study Date': *}
            pat2Proc is a dictionary containing patient info
            subID is the name of the subjects directory (basically sub-<accession number>)
            subDir is the absolute path to subID
    
    print: all exams without processing input in /data/Patients/tmp_dicom
    
    """
    
    accList = []
    for folder in os.listdir(wDir):
        if folder.startswith('sub-'):
            accNumber = folder.split('sub-')[-1]
            accList.append(accNumber)
        
    newPatList = []
    notProcessedList = []
    
    patList = [x for x in os.scandir(patDcmRoot) if x.is_dir()]
    for i, folder in enumerate(patList):
        dcmTags = get_dicom_info(folder)
        newPatList.append(dcmTags)
        
    if sortcriterion == 'name':
        newPatList = sorted(newPatList, key=lambda k: k['Patient Name'])
    elif sortcriterion == 'date':
        newPatList = sorted(newPatList, key=lambda k: k['Study Date'])
    
    for item in newPatList:
        if item['Accession Number'] not in accList:
            notProcessedList.append(item)
    
    if len(notProcessedList) == 0:
        print('No more patients to process.')
        return
    else:
        print('List of patient exams in need of processing:')
        print('')
        for i, item in enumerate(notProcessedList):
            print('%i. %s (%s) - datum: %s' %(i+1, item['Patient Name'], item['Patient ID'], item['Study Date']))
            
    # Select the patient dicom directory by number
    patDirNum = int(input('Number of the patient dicom directory to process: '))

    # Check for number in correct range and loop untill correct
    while patDirNum > len(notProcessedList):
        print('Select a number in the correct range (1 - ' + str(len(notProcessedList)) + ')!')
        patDirNum = int(input('Try again! Number of the patient dicom directory to process: '))
    else:
        pat2Proc = notProcessedList[patDirNum-1]
        patName = pat2Proc['Patient Name']
        patAdrema = pat2Proc['Patient ID']
        patDate = pat2Proc['Study Date']
        patID = 'sub-' + pat2Proc['Accession Number']

        print('The selected patient = %s, ID is %s, exam date = %s.' % (patName, patAdrema, patDate))
        print('Subject directory = %s/%s' % (wDir, patID))
        
        
    subID = 'sub-' + pat2Proc['Accession Number']
    subDir = opj(wDir, subID)
    
    return notProcessedList, pat2Proc, subID, subDir


# ### Create JSON log file

# In[13]:


# Start the log file for the new subject
def create_json_log_file():
    """
    Function to create the log.json file and fill in the fields to start with
    
    subID = subject's directory name under /data/Patients/
    
    returns subLogFile
        
    """
  
    os.chdir(opj(wDir, subID))
    
    # Construct the name of the log file
    logFileName = subID + '_log.json'
    subLogFile = opj(wDir, iDir, logFileName)
    
    if os.path.isfile(opj(iDir, logFileName)):
        print('Log file(s) exist in %s: %s' % (subID, logFileName))
        print('The file %s contains this info:' % subLogFile)
        with open(subLogFile, "r") as f:
            jData = json.load(f)
            pprint.pprint(jData)
    else:
        referral = input("Patient referred for: ")
        
        shutil.copyfile(opj(bidsSource, 'subject_log.json'), subLogFile)
            
        with open(subLogFile, "r") as jFile:
            jData = json.load(jFile)
        
        tmp = jData["Subdir_creation"]
        jData["Subdir_creation"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        jData["Referral"] = referral
        jData['Subject DCM tags'] = pat2Proc
            
        with open(subLogFile, "w") as jFile:
            json.dump(jData, jFile)
        
        print('New log file: %s' % subLogFile) 
        
    return subLogFile


# ### dcm2niftiix

# In[14]:


def dcm2niftii():
    """
    Function to start the convertion of dicom to niftii
    pat2Proc is defined
    
    """
    
    for dcmDir in os.listdir(patDcmRoot):
        if stripaccents(pat2Proc['Patient Name'].lower().replace(' ', '_')) in dcmDir.lower():
            convertDir = opj(patDcmRoot, dcmDir)
    
    converter = Dcm2niix()
    converter.inputs.source_dir = convertDir
    converter.inputs.compression = 5
    converter.inputs.output_dir = os.path.join(wDir, subID)
        
    converter.run()


# In[15]:


#stripaccents(pat2Proc['Patient Name'].lower().replace(' ', '_'))


# In[16]:


def f_series_numbers(fDir):
    """
    function to find the right series after dcm2niix conversion
    function also finds the number of images in the associated nii.gz files
    
    subDir : the absolute path to the subject dir
    fDict : dictionary linking SeriesDescriptions -> subdir
    pat2Proc is defined
    
    returns sDict, a dictionary with series numbers as keys and subdirs as value
    
    """
    
    sDict = {}
    
    for ff in os.listdir(fDir):
        if ff.endswith('.json'):
            with open(opj(fDir, ff), 'r') as f:
                jsonData = f.read()
                jData = json.loads(jsonData)
                sDescript = jData['SeriesDescription'].lower()
                if 'fused' not in sDescript:
                    for jVal in fDict.keys():
                        if jVal in sDescript and 'mpr_' not in sDescript:
                            sDict[ff] = fDict[jVal]
                        
    return sDict


# In[17]:


def f_series_timepoints(fDir):
    """
    function to check for the number of timepoints in the fMRI series 
     
    subDir : the absolute path to the dir with the files
    sDict: the dictionary with series numbers as keys and subdirs as value
    pat2Proc is defined
    
    returns pDict, a dictionary with series descriptions as keys and number of images (time-points) as value
    
    """

    pDict = {}
    
    for jsonFile in sDict:
        niigzFile  = jsonFile.split('.')[0]+'.nii.gz'
        niiImage = nib.load(opj(fDir, niigzFile))
        niiHeader = niiImage.header
        funcDim = niiHeader['dim'][4]
        pDict[jsonFile] = funcDim

    return pDict


# ### copy2dirs

# In[18]:


def copy2dirs():
    """
    Function to show the number of time points in functional runs
    If not correct, the option to quit is provided
    pat2Proc, sDict and pDict have to be defined
    
    """
    
    print('Volgende series zijn in overweging genomen:')
    
    for jsonFile in sDict:
        numImages = pDict[jsonFile]
        for item in list(fimgDict.keys()):
            if item in jsonFile and pDict[jsonFile] in fimgDict[item]:
                print('%s moet naar %s (aantal tijdspunten is %i = correct)' %(jsonFile, sDict[jsonFile], numImages))
            elif item in jsonFile and pDict[jsonFile] not in fimgDict[item]:
                print('%s moet naar %s (slechts %i tijdspunten: fout)' %(jsonFile, 'not_used', numImages))
                if yes_or_no('Toch behouden?') == True:
                    print('OK')
                else:
                    print('Trash')
                    #del pDict[jsonfile]
    
    print('- Alle andere series worden gecopiëerd naar map "not_used".\n')
    
    if yes_or_no('Do you want to continue?\n') == True:
        print('Copying files....')

        for jsonFile in sDict:
            numImages = pDict[jsonFile]
            for item in list(fimgDict.keys()):
                if item in jsonFile and pDict[jsonFile] in fimgDict[item]:
                    niigzFile = jsonFile.split('.')[0] + '.nii.gz'
                    shutil.move(opj(subDir, jsonFile), opj(subDir, sDict[jsonFile], jsonFile))
                    shutil.move(opj(subDir, niigzFile), opj(subDir, sDict[jsonFile], niigzFile))
                elif item in jsonFile and pDict[jsonFile] not in fimgDict[item]:
                    niigzFile = jsonFile.split('.')[0] + '.nii.gz'
                    shutil.move(opj(subDir, jsonFile), opj(subDir, 'not_used', jsonFile))
                    shutil.move(opj(subDir, niigzFile), opj(subDir, 'not_used', niigzFile))

        for ff in os.listdir(opj(subDir)):
            if ff != subID + '_log.json' and os.path.isfile(opj(subDir, ff)):
                shutil.move(opj(subDir, ff), opj(subDir, 'not_used', ff))
                


# In[19]:


def newcopy2dirs():
    """
    Function to show the number of time points in functional runs
    If not correct, the option to quit is provided
    pat2Proc, sDict and pDict have to be defined
    
    """
    
    newpDict = pDict
    
    print('Volgende series zijn in overweging genomen:')
    
    for jsonFile in sDict:
        numImages = newpDict[jsonFile]
        for item in list(fimgDict.keys()):
            if item in jsonFile and newpDict[jsonFile] in fimgDict[item]:
                print('%s moet naar %s (aantal tijdspunten is %i = correct)' %(jsonFile, sDict[jsonFile], numImages))
            elif item in jsonFile and newpDict[jsonFile] not in fimgDict[item]:
                print('%s moet naar %s (slechts %i tijdspunten: fout)' %(jsonFile, 'not_used', numImages))
                if yes_or_no('Toch behouden?') == True:
                    print('OK')
                else:
                    print('Trash')
                    del newpDict[jsonFile]
    
    print('- Alle andere series worden gecopiëerd naar map "not_used".\n')
    
    if yes_or_no('Do you want to continue?\n') == True:
        print('Copying files....')
    
        for jsonFile in sDict:
            if jsonFile in newpDict:
                numImages = newpDict[jsonFile]
                niigzFile = jsonFile.split('.')[0] + '.nii.gz'
                shutil.move(opj(subDir, jsonFile), opj(subDir, sDict[jsonFile], jsonFile))
                shutil.move(opj(subDir, niigzFile), opj(subDir, sDict[jsonFile], niigzFile))
            else:
                niigzFile = jsonFile.split('.')[0] + '.nii.gz'
                shutil.move(opj(subDir, jsonFile), opj(subDir, 'not_used', jsonFile))
                shutil.move(opj(subDir, niigzFile), opj(subDir, 'not_used', niigzFile))                

        for ff in os.listdir(opj(subDir)):
            if ff != subID + '_log.json' and os.path.isfile(opj(subDir, ff)):
                shutil.move(opj(subDir, ff), opj(subDir, 'not_used', ff))
    
    return newpDict


# ### show_asl

# In[20]:


def show_asl():
    """
    Function to detect and copy ASL sequences and copy these the 'asl' directory
    
    """
    
    # Sort possible ASL files in lists
    
    PCASLList = []
    M0List = []
    PASLList = []
    multiTIList = [] 
    
    for ff in os.listdir(opj(wDir, subID, 'not_used')):
        if 'pcasl' in ff.lower():
            PCASLList.append(ff)
        elif 'm0' in ff.lower():
            M0List.append(ff)
        elif 'pasl' in ff.lower():
            PASLList.append(ff)
        elif 'multi' in ff.lower():
            multiTIList.append(ff)
    
    for ff in PCASLList:
        if 'm0' in ff.lower():
            PCASLList.remove(ff)
            M0List.append(ff)
    
    print('Volgende ASL gerelateerde series zijn gevonden:')
        
    if len(PCASLList) == 0:
        print('\nNo pCASL files.')
    else:
        print('\npCASL files:')
        for f in PCASLList:
            if f.endswith('.json'):
                print(' - %s' % f.split('.json')[0])
            
    if len(M0List) == 0:
        print('\nNo M0 files.')
    else:
        print('\nM0 files:')
        for f in M0List:
            if f.endswith('.json'):
                print(' - %s' % f.split('.json')[0])
            
    if len(PASLList) == 0:
        print('\nNo PASL files.')
    else:
        print('\nPASL files:')
        for f in PASLList:
            if f.endswith('.json'):
                print(' - %s' % f.split('.json')[0])
            
    if len(multiTIList) == 0:
        print('\nNo multiTI files.')
    else:
        print('\nMultiTI files:')
        for f in multiTIList:
            if f.endswith('.json'):
                print(' - %s' % f.split('.json')[0])
            
    allASLList = PCASLList + M0List + PASLList + multiTIList
            
    if yes_or_no('\nOK to copy?'):
        for f in allASLList:
            shutil.move(opj(wDir, subID, 'not_used', f), opj(wDir, subID, 'asl', f))
        print('ASL files were moved to asl directory.')
    else:
        print('No files were moved to asl directory')


# ### Create taskList - dwi2waste

# In[21]:


def cre_task_list():
    """
    create the task list from the files in the func dir
    write paradigms to json file of subject
    
    pat2Proc is defined
    
    returns: taskList
    
    """

    funcDir = opj(wDir, subID, 'func')
    
    taskList =[]
    for ff in os.listdir(funcDir):
        dotPos = ff.find('.')
        if ff[dotPos+1: ] == 'json':
            for key in fDict.keys():
                if key in ff.lower() and key.lower() != 't1_mprage':
                    key = key.replace('_', '')
                    taskList.append(key.lower())
                    
    # Write the paradigms in func to the log file
    logfile2Update = opj(iDir, subID + '_log.json')

    with open(logfile2Update, "r") as jFile:
            jData = json.load(jFile)

    jData["Paradigms"] = taskList

    with open(logfile2Update, "w") as jFile:
            json.dump(jData, jFile)

    print(json.dumps(jData, indent=4, sort_keys=True))
    for i, task in enumerate(taskList):
        task = re.sub('_', '', task)
        taskList[i] = task
        
    return taskList


# In[22]:


def dwi2waste():
    """
    Function to get rid of unnecessary dwi files
    
    dwiPath = full path to directory where dwi niftii files are stored
    
    """
    
    dwiPath = opj(wDir, subID, 'dwi')
    
    # if there are no dwi files, exit
    if len(os.listdir(dwiPath)) == 0:
        print('No dwi files')
        return
    
    dwiDict = {}
    for ff in os.listdir(dwiPath):
        if ff.endswith('.json'):
            with open(opj(dwiPath, ff)) as jFile:
                jData = json.load(jFile)
                dwiDescript = str(jData['SeriesDescription'].split('_')[-1])
                sNum = ff.split('.')[0].split('_')[-1]
                dwiDict[dwiDescript] = sNum
    
    print(dwiDict)
  
    # move non relevant files to not_used
    for ff in os.listdir(dwiPath):
        if ff.split('.')[0].endswith(dwiDict['sms']):
            pass
        else:
            os.rename(opj(dwiPath, ff), os.path.join(subDir, "not_used", ff))  


# ### Rename functions

# In[23]:


def rename_anat():
    """ 
    Rename the files output by dcm2niix in the anat folder to BIDS
    mprage, flair3d and swi images are taken into account
    subID and subDir are defined
    
    """
    
    if len(os.listdir(opj(subDir, 'anat'))) == 0:
        print("No files in %s" % opj(subDir, 'anat'))
    else:
        # T1w, FLAIR and T2star are possible BIDS names
        anatFileListPre = os.listdir(opj(subDir, 'anat'))
        for ff in anatFileListPre:
            
            if 'mprage' in ff.lower() and ff.endswith('json'):
                print('T1 files present and renamed.')
                fff = ff.split('.')[0]
                os.rename(opj(subDir, 'anat', fff + '.nii.gz'), opj(subDir, 'anat', subID + '_T1w.nii.gz'))
                os.rename(opj(subDir, 'anat', fff + '.json'), opj(subDir, 'anat', subID + '_T1w.json'))
                
            elif 'flair3d' in ff.lower() and ff.endswith('json'):
                print('FLAIR3D files present and renamed')
                fff = ff.split('.')[0]
                os.rename(opj(subDir, 'anat', fff + '.nii.gz'), opj(subDir, 'anat', subID + '_FLAIR.nii.gz'))
                os.rename(opj(subDir, 'anat', fff + '.json'), opj(subDir, 'anat', subID + '_FLAIR.json'))
                
            elif 'swi' in ff.lower() and ff.endswith('json'):
                with open(opj(subDir, 'anat', ff), 'r') as jFile:
                    jData = json.load(jFile)
                    if 'SWI' in jData['ImageType']:
                        print('SWI files present and renamed')
                        fff = ff.split('.json')[0]
                        os.rename(opj(subDir, 'anat', fff + '.nii.gz'), opj(subDir, 'anat', subID + '_T2star.nii.gz'))
                        os.rename(opj(subDir, 'anat', fff + '.json'), opj(subDir, 'anat', subID + '_T2star.json'))
                    elif 'MNIP' in jData['ImageType']:
                        print('Removing SWI MIP images to "not_used" directory')
                        fff = ff.split('.json')[0]
                        shutil.move(opj(subDir, 'anat', fff + '.json'), opj(subDir, 'not_used', fff + '.json'))
                        shutil.move(opj(subDir, 'anat', fff + '.nii.gz'), opj(subDir, 'not_used', fff + '.nii.gz'))


# In[24]:


def rename_func():
    """ 
    Rename the files output by dcm2niix in the func folder to BIDS
    Write task to json file
    subID and subDir are defined
    Horrible code ...
    
    """
        
    if len(os.listdir(opj(subDir, 'func'))) == 0:
        print("No files in %s" % opj(subDir, 'func'))
    else:
        funcFileListPre = os.listdir(opj(subDir, 'func'))
        funcFileList = [x.split('.')[0] for x in funcFileListPre if x.endswith('.json')]
        rsFileList = [x for x in funcFileList if 'restingstate' in x.lower()]
        taskFileList = [x for x in funcFileList if x not in rsFileList]
        for item in fDict.keys():
            for fName in taskFileList:
                if item in fName.lower():
                    if '_' in item: 
                        itm = re.sub('_', '', item)
                    else:
                        itm = item
                    print('Changing %s -> %s in func' %(fName, subID + '_task-' + itm + '_bold'))
                    os.rename(opj(subDir, 'func', fName + '.nii.gz'), opj(subDir, 'func', subID + '_task-' + itm + '_bold.nii.gz'))
                    os.rename(opj(subDir, 'func', fName + '.json'), opj(subDir, 'func', subID + '_task-' + itm + '_bold.json'))
                    print('Adding "TaskName":"%s" to %s.json' %(itm, subID + '_task-' + itm + '_bold'))
                    with open(opj(subDir, 'func', subID + '_task-' + itm + '_bold.json'), "r") as jFile:
                        jData = json.load(jFile)
                        try:
                            TaskName = jData['TaskName']
                        except:
                            jData['TaskName'] = itm
                    with open(opj(subDir, 'func', subID + '_task-' + itm + '_bold.json'), "w") as jFile:
                        json.dump(jData, jFile)
                    print('Adding %s file for events' % opj(subDir, 'func', subID + '_task-' + itm + '_events.tsv'))
                    shutil.copyfile(opj(bidsSource, itm + '_seconds.tsv'), opj(subDir, 'func', subID + '_task-' + itm + '_events.tsv'))
        if len(rsFileList) > 1:
            for idx, item in enumerate(rsFileList):
                os.rename(opj(subDir, 'func', item + '.nii.gz'), opj(subDir, 'func', subID + '_rest_run-' + str(idx+1) + '_bold.nii.gz'))
                os.rename(opj(subDir, 'func', item + '.json'), opj(subDir, 'func', subID + '_rest_run-' + str(idx+1) + '_bold.json'))
                print('Changing %s -> %s in func' %(item, subID + '_rest_run-' + str(idx+1) + '_bold'))
        elif len(rsFileList) == 1:
            os.rename(opj(subDir, 'func', rsFileList[0] + '.nii.gz'), opj(subDir, 'func', subID + '_rest_bold.nii.gz'))
            os.rename(opj(subDir, 'func', rsFileList[0] + '.json'), opj(subDir, 'func', subID + '_rest_bold.json'))                    
            print('Changing %s -> %s in func' %(item, subID + '_rest_bold'))


# In[25]:


def newrename_func():
    """ 
    Rename the files output by dcm2niix in the func folder to BIDS
    Write task to json file
    subID and subDir are defined
    Horrible code ...
    
    """
    
    funcDict = {}
    
    if len(os.listdir(opj(subDir, 'func'))) == 0:
        print("No files in %s" % opj(subDir, 'func'))
    else:
        for ff in os.listdir(opj(subDir, 'func')):
            if ff.endswith('json'):
                for item in fDict:
                    if item in ff.lower():
                        funcDict[ff] = [newpDict[ff], item]
                        
    for ffunc in funcDict:
        rsFileList = [x.split('.')[0] for x in list(funcDict) if 'restingstate' in x.lower()]
        taskFileList = [x.split('.')[0] for x in list(funcDict) if x not in rsFileList]
    
    #print(rsFileList)
    #print(taskFileList)

    if len(rsFileList) > 1:
        for idx, item in enumerate(rsFileList):
            os.rename(opj(subDir, 'func', item + '.nii.gz'), opj(subDir, 'func', subID + '_rest_run-' + str(idx+1) + '_bold.nii.gz'))
            os.rename(opj(subDir, 'func', item + '.json'), opj(subDir, 'func', subID + '_rest_run-' + str(idx+1) + '_bold.json'))
            print('Changing %s -> %s in func' %(item, subID + '_rest_run-' + str(idx+1) + '_bold'))
    elif len(rsFileList) == 1:
        os.rename(opj(subDir, 'func', rsFileList[0] + '.nii.gz'), opj(subDir, 'func', subID + '_rest_bold.nii.gz'))
        os.rename(opj(subDir, 'func', rsFileList[0] + '.json'), opj(subDir, 'func', subID + '_rest_bold.json'))                    
        print('Changing %s -> %s in func' %(item, subID + '_rest_bold'))
        
    # code to rename tasks with the correct number of EPI volumes and runs
    
    return funcDict


# In[26]:


def rename_fmap():
    """ 
    Rename the files output by dcm2niix in the fmap folder to BIDS
    Write TE's to json file
    subject = subjects directory in wDir
    
    """

    fmapDict = {'e1':'magnitude1', 'e2':'magnitude2', 'ph':'phasediff'}
        
    if len(os.listdir(opj(subDir, 'fmap'))) != 6:
        print("Number of files in %s different from mandatory 6!" % opj(subDir, 'fmap'))
    else:
        fmapPreFileList = os.listdir(opj(subDir, 'fmap'))
        fmapFileListJson = [x.split('.')[0] for x in fmapPreFileList if x.endswith('.json')]
        for ff in fmapFileListJson:
            res = [x for x in fmapDict.keys() if(x in ff)]
            print('Changing %s -> %s in file names' %(ff, subID + '_' + fmapDict[res[-1]]))
            os.rename(opj(subDir, 'fmap', ff + '.json'), opj(subDir, 'fmap', subID + '_' + fmapDict[res[-1]] + '.json'))
            os.rename(opj(subDir, 'fmap', ff + '.nii.gz'), opj(subDir, 'fmap', subID + '_' + fmapDict[res[-1]] + '.nii.gz'))
        print('Adding EchoTime1 and EchoTime2 to %s' % subID + '_phasediff.json')
        with open(opj(subDir, 'fmap', subID + '_magnitude1.json'), "r") as j1File:
            jData = json.load(j1File)
            TE1 = jData['EchoTime']
        with open(opj(subDir, 'fmap', subID + '_magnitude2.json'), "r") as j2File:
            jData = json.load(j2File)
            TE2 = jData['EchoTime']        
        with open(opj(subDir, 'fmap', subID + '_phasediff.json'), "r") as jphFile:
            jData = json.load(jphFile)
            jData['EchoTime1'] = TE1
            jData['EchoTime2'] = TE2
        with open(opj(subDir, 'fmap', subID + '_phasediff.json'), "w") as jphFile:
            json.dump(jData, jphFile)


# In[27]:


def rename_dwi():
    """ 
    Rename the files output by dcm2niix in the dwi folder to BIDS
    TODO
    subject = subjects directory in wDir
    
    """
    
    dwiFileList = os.listdir(opj(subDir, 'dwi'))
    
    if len(dwiFileList) == 0:
        print("No files in %s" % opj(subDir, 'dwi'))
    else:
        for idx, ff in enumerate(dwiFileList):
            if ff.endswith('nii.gz'):
                print('rename %s -> %s' %(ff, subID + '_dwi.nii.gz'))
                os.rename(opj(subDir, 'dwi', ff), opj(subDir, 'dwi', subID + '_dwi.nii.gz'))
            elif ff.endswith('bval'):
                print('rename %s -> %s' %(ff, subID + '_dwi.bval'))
                os.rename(opj(subDir, 'dwi', ff), opj(subDir, 'dwi', subID + '_dwi.bval'))
            elif ff.endswith('bvec'):
                print('rename %s -> %s' %(ff, subID + '_dwi.bvec'))
                os.rename(opj(subDir, 'dwi', ff), opj(subDir, 'dwi', subID + '_dwi.bvec'))
            elif ff.endswith('json'):
                print('rename %s -> %s' %(ff, subID + '_dwi.json'))
                os.rename(opj(subDir, 'dwi', ff), opj(subDir, 'dwi', subID + '_dwi.json'))


# In[28]:


def rename_asl():
    """
    Rename the files in de 'asl' deriectory
    Not quite BIDS yet
    
    convention
    sub-<label>[_technique][_PA/AP][_single/multi][_ss/ns/di][_PLD/TI][_run<#>].<nii.gz/json>
    technique = [PCASL, PASL, M0]
    
    """
    
    aslFileList = [x.split('.json')[0] for x in os.listdir(opj(subDir, 'asl')) if x.endswith('json')]
    pcaslList = [x for x in aslFileList if 'pcasl' in x.lower()]
    # Get rid of any M0 images in pcaslList
    pcaslList = [x for x in pcaslList if 'm0' not in x.lower()]
    paslList = [x for x in aslFileList if 'pasl' in x.lower()]
    m0List = [x for x in aslFileList if 'm0' in x.lower()]
    restList = [x for x in aslFileList if x not in (pcaslList+paslList+m0List)]
    
    seriesDescrList = []
    
    # First rename the M0 images    
    if len(m0List) != 0:
        print('There are %i M0 images' % len(m0List))
        seriesDescrList = []
        m0Dict = {}
        for fName in m0List:
            #with open(opj(subDir, 'asl', fName + '.json'), 'r') as jFile:
            #    jData =  json.load(jFile)
            
            seriesDescr = 'unknown'
            
            if '_PA_' in fName:
                seriesDescr = 'PA'
            elif '_AP_' in fName:
                seriesDescr = 'AP'
            #seriesDescr = jData['SeriesDescription']
            if seriesDescr not in seriesDescrList:
                seriesDescrList.append(seriesDescr)
                m0Dict[seriesDescr] = [fName]            
            else:
                m0Dict[seriesDescr] = m0Dict[seriesDescr] + [fName]
            
        for m0Direction in m0Dict.keys():
            for idx, ff in enumerate(sorted(m0Dict[m0Direction])):
                newName = subID + '_M0_' + m0Direction + '_run-' + str(idx+1)
                print('Rename %s -> %s' % (ff, newName))
                os.rename(opj(subDir, 'asl', ff + '.json'), opj(subDir, 'asl', newName + '.json'))
                os.rename(opj(subDir, 'asl', ff + '.nii.gz'), opj(subDir, 'asl', newName + '.nii.gz')) 
    
    # Make a list of pCASL contrasts if there are pCASL files
    
    if len(pcaslList) != 0:
        
        seriesDescrList = []
        pcaslDict = {}
        for fName in sorted(pcaslList):
            with open(opj(subDir, 'asl', fName + '.json'), 'r') as jFile:
                jData =  json.load(jFile)
            seriesDescr = jData['SeriesDescription']
            if seriesDescr not in seriesDescrList:
                seriesDescrList.append(seriesDescr)
                pcaslDict[seriesDescr] = [fName]
            else:
                pcaslDict[seriesDescr] = pcaslDict[seriesDescr] + [fName]
        
        print('There are %i pCASL images:' % len(pcaslList))
        
        for pcaslKey in pcaslDict.keys():
            for idx, ff in enumerate(pcaslDict[pcaslKey]):
                newName = subID + '_PCASL' + '_single-' + pcaslKey.split('_')[-1] + '_' + pcaslKey.split('_')[0]  + '_run-' + str(idx+1)
                print('Rename %s -> %s' % (ff, newName))
                os.rename(opj(subDir, 'asl', ff + '.json'), opj(subDir, 'asl', newName + '.json'))
                os.rename(opj(subDir, 'asl', ff + '.nii.gz'), opj(subDir, 'asl', newName + '.nii.gz'))                
                
    if len(paslList) != 0:
        print('There are %i PASL images' % len(paslList))
        for ff in sorted(paslList):
            with open(opj(subDir, 'asl', ff + '.json'), 'r') as jFile:
                jData = json.load(jFile)
            print('%s -> %s_PASL_multi-%s_%s' %(ff, subID, jData['SeriesDescription'].split('_')[2], jData['SeriesDescription'].split('_')[0]))
            newName = subID+'_PASL_multi-'+jData['SeriesDescription'].split('_')[2]+'_'+jData['SeriesDescription'].split('_')[0]
            os.rename(opj(subDir, 'asl', ff + '.json'), opj(subDir, 'asl', newName + '.json'))
            os.rename(opj(subDir, 'asl', ff + '.nii.gz'), opj(subDir, 'asl', newName + '.nii.gz'))  


# ### Slice timing

# In[29]:


def get_param_slicetiming(task):
    """
    function to extract TR, slice order, number of slices and ref slice number for the functional images
    
    :param: the paradigm, of which there is always a paradigm.json file associated
    :param type: text
    
    :returns: the list slicetiming_param = [TR, number of slices, ref slice number, [slice order]].
    
    """
  
    slicetimingParam = []
    headerFMRI = []
    
    jsonFile = opj(subDir, 'func', subID + '_task-' + task + '_bold.json')
    fmriFile = opj(subDir, 'func', subID + '_task-' + task + '_bold.nii.gz')
  
    with open(jsonFile, 'r') as ff:
        jFile=ff.read()
        jData = json.loads(jFile)

        TR = jData["RepetitionTime"]
        try:
            timing = jData["SliceTiming"]
        except:
            headerFMRI = get_ipython().getoutput("fslhd '{fmriFile}'")
            numSlices = (headerFMRI[7]).replace('\t', ' ').split(' ')[-1]
            timing = [x*0.01 for x in range(0, int(numSlices))]
        
        # initialize slice_order
        sliceOrder = [0]*len(timing)

        # First we have to find out if the sequence used SMS, values will appear more than once

        smsFactor = len(timing) / len(sorted(set([i for i in timing if timing.count(i)>=1])))
        smsSlices = int(len(timing) / smsFactor)
        smsTiming = ([timing[i:i + smsSlices] for i in range(0, len(timing), smsSlices)])[0]
        smsSliceOrder = [0]*len(smsTiming)

        if smsFactor == 1:
            for i, x in enumerate(sorted(list(range(len(timing))), key=lambda y: timing[y])):
                sliceOrder[x] = i + 1
        else:
            for i, x in enumerate(sorted(list(range(len(smsTiming))), key=lambda y: smsTiming[y])):
                smsSliceOrder[x] = i + 1

        if sum(sliceOrder) != 0:
            refSl = np.argmin(sliceOrder) + 1

        if sum(smsSliceOrder) != 0:
            sliceOrder = 4 * smsSliceOrder
            refSl = np.argmin(smsSliceOrder) + 1

        nSlices = len(sliceOrder)

        slicetimingParam.append(TR)
        slicetimingParam.append(nSlices)
        slicetimingParam.append(refSl)
        slicetimingParam.append(sliceOrder)
    
    return slicetimingParam


# ### Templates list

# In[30]:


def maketemplateslist():
    """
    Function to populate the templatesList
    SubID and subDir have to be defined
    taskList has to be defined
    
    """
   
    # Choose the anatomy file
    for i, ff in enumerate(os.listdir(opj(subDir, 'anat'))):
        if ff.endswith('json'):
            print(ff.split('.')[0])
        
    aVar = input('Select the anatomy variety? [default = T1]')
    
    if aVar in ['flair', 'FLAIR', 'flair3d', 'FLAIR3D']:
        try:
            anatImg = opj(subDir, 'anat', subID + '_FLAIR.nii.gz')
        except:
            print('No FLAIR3D image found!')
    else:
        try:
            anatImg = opj(subDir, 'anat', subID + '_T1w.nii.gz')
        except:
            print('No 3D T1 image found')

    templatesList = []
    
    for i, task in enumerate(taskList):
        templatesList.append('templates_' + task)
        templatesList[i] = {'anat': anatImg,
                            'func': opj(subDir, 'func', subID + '_task-' + task + '_bold.nii.gz')}
    
    # check if the files exist
    
    nonExistingFileList = []
    
    for i, item in enumerate(templatesList):
        for key in templatesList[i].keys():
            ff = templatesList[i][key]
            myFf = Path(ff)
            try:
                ff = myFf.resolve()
            except:
                nonExistingFileList.append(myFf)
                
    if len(nonExistingFileList) == 0:
        print('All files are present!')
    else:
        print('The following files where not found: %s' % nonExistingFileList)
        
    if yes_or_no('Do you want to continue?') == True:
        pass
    else:
        quit()
        
    return templatesList, anatImg


# ## Processing functions

# In[31]:


# Tell fsl to generate all output in uncompressed nifti format
fsl.FSLCommand.set_default_output_type('NIFTI')

# Set the way matlab should be called
matlb.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# In case a different path is required
# mlab.MatlabCommand.set_default_paths('/software/matlab/spm12b/spm12b_r5918')


# ### Preprocessing

# In[32]:


def preproc_fMRI_UZG(ST = 'Y'):
    """
    This function organizes the workflow for preprocessing
    subID = the definition of the subject in /data/Patients (sub-<accession number>)
    
    pat2Proc, templatesList are defined
    
    """
    
    gunzipFuncList = []
    slicetimingParamList = []
    sfList = []
    slicetimeList = []
    realignList = []
    mcflirtList = []
    artList = []
    coregList = []
    applywarpList = []
    smoothList = []
    maskFuncList = []
    detrendList = []
    normFuncList = []
    gunzipDetrendList = []
    
    # Initiate Gunzip nodes

    gunzip_anat = pe.Node(interface = Gunzip(in_file = templatesList[0]['anat']), name = 'gunzip_anat')
    
    for i, task in enumerate(taskList):
        gunzipName = 'gunzip_' + task
        gunzipFuncList.append(gunzipName)
        gunzipFuncList[i] = pe.Node(interface = Gunzip(in_file = templatesList[i]['func']), name= gunzipName)
        slicetimingParam = get_param_slicetiming(task)
        slicetimingParamList.append(slicetimingParam)

    # Initiate the datasink node
    outputFolder = 'results_' + subID + '-N' # all resulting images in MNI space
    datasink = pe.Node(interface = nio.DataSink(base_directory=oDir,
                             container=outputFolder), name='datasink')
    
    # Create SelectFiles nodes
    for i, task in enumerate(taskList):
        selectFileTask = 'sf_' + task
        sfList.append(selectFileTask)
        sfList[i] = pe.Node(interface = SelectFiles(templatesList[i],
                                                    base_directory=wDir,
                                                    sort_filelist=True), 
                            name = selectFileTask)
                              
    # Define the nodes

    # Initiate the segment node here: ONLY ANATOMY
    segment = pe.Node(interface = spm.NewSegment(tissues=tissues), name='segment')

    # Threshold - Threshold WM probability image: ONLY ANATOMY
    threshold_WM = pe.Node(interface = fsl.Threshold(thresh=0.5,
                                                     args='-bin',
                                                     output_type='NIFTI'), 
                                                     name="threshold_WM")

    # Initiate resample node: ONLY ANATOMY
    resample = pe.Node(interface = fsl.FLIRT(apply_isoxfm=desiredVoxelIso,
                                             output_type='NIFTI'), 
                                             name="resample")

    # Threshold - Threshold GM probability image: ONLY ANATOMY
    mask_GM = pe.Node(interface = fsl.Threshold(thresh=0.5,
                                                args='-bin -dilF',
                                                output_type='NIFTI'), 
                                                name="mask_GM")

    # Normalize anatomy: ONLY ANATOMY
    norm_anat = pe.Node(interface = spm.Normalize12(jobtype='estwrite',
                                                    write_voxel_sizes=[1, 1, 1]), 
                                                    name='norm_anat')
                              
    for i, task in enumerate(taskList):
        slicetimeList.append('slicetime_' + task)
        slicetimeList[i] = pe.Node(interface = spm.SliceTiming(num_slices=slicetimingParamList[i][1],
                                                               ref_slice=slicetimingParamList[i][2],
                                                               slice_order=slicetimingParamList[i][3],
                                                               time_repetition=slicetimingParamList[i][0],
                                                               time_acquisition=slicetimingParamList[i][0]-(slicetimingParamList[i][0]/slicetimingParamList[i][1])), 
                                                               name='slicetime_' + task)
        realignList.append('realign_' + task)
        realignList[i] = pe.Node(interface = spm.Realign(register_to_mean=True,
                                                         quality=0.7),
                                                         name='realign_' + task)
        artList.append('art_' + task)
        artList[i] = pe.Node(interface = ra.ArtifactDetect(norm_threshold=1,
                                                            zintensity_threshold=3,
                                                            mask_type='spm_global',
                                                            parameter_source='SPM',
                                                            use_differences=[True, False],
                                                            plot_type='svg'), 
                                                            name='art_' + task)
        coregList.append('coreg_' + task)
        coregList[i] = pe.Node(interface = fsl.FLIRT(dof=6,
                                                      cost='bbr',
                                                      schedule=bbrPath,
                                                      output_type='NIFTI'), 
                                                      name='coreg_' + task)
        applywarpList.append('applywarp_' + task)
        applywarpList[i] = pe.Node(interface = fsl.FLIRT(interp='spline',
                                                          apply_isoxfm=desiredVoxelIso,
                                                          output_type='NIFTI'), 
                                                          name='applywarp_' + task)
        smoothList.append('smooth_' + task)
        smoothList[i] = pe.Node(interface = spm.Smooth(fwhm=6), name = 'smooth_' + task)
        maskFuncList.append('mask_func_' + task)
        maskFuncList[i] = pe.MapNode(interface = fsl.ApplyMask(output_type='NIFTI'),
                                     name='mask_func', 
                                     iterfield=['in_file'])
        detrendList.append('detrend_' + task)
        detrendList[i] = pe.Node(interface = cnfds.TSNR(regress_poly=2), name='detrend_' + task)
        gunzipDetrendList.append('gunzip_detrend_' + task)
        gunzipDetrendList[i] = pe.Node(interface = Gunzip(), name='gunzip_detrend_' + task)
        normFuncList.append('detrend_norm_' + task)
        normFuncList[i] = pe.Node(interface = spm.Normalize12(jobtype='write',
                                                              write_voxel_sizes=[3, 3, 3]),
                                                              name='normalized_' + task)
    
    # Connect the workflow
    preprocList = []

    suboDir = opj(oDir, subID)
    
    if ST == 'Y':
        # workflow with slice timing
        for i, task in enumerate(taskList):
            preprocList.append('preproc_' + task)
            preprocList[i] = pe.Workflow(name='work_preproc_' + task, base_dir=suboDir)
            preprocList[i].connect([(sfList[i], gunzip_anat, [('anat', 'in_file')]),
                                     (sfList[i], gunzipFuncList[i], [('func', 'in_file')]),
                                     (gunzipFuncList[i], slicetimeList[i], [('out_file', 'in_files')]),
                                     (slicetimeList[i], realignList[i], [('timecorrected_files', 'in_files')]),
                                     (realignList[i], artList[i], [('realigned_files', 'realigned_files'),
                                                                     ('realignment_parameters', 'realignment_parameters')]),
                                     (gunzip_anat, segment, [('out_file', 'channel_files')]),
                                     (gunzip_anat, norm_anat, [('out_file' , 'image_to_align')]),
                                     (norm_anat, datasink, [('normalized_image', 'preproc.@wanat')]),
                                     (gunzip_anat, coregList[i], [('out_file', 'reference')]),
                                     (realignList[i], coregList[i], [('mean_image', 'in_file')]),
                                     (segment, threshold_WM, [(('native_class_images', get_wm), 'in_file')]),
                                     (threshold_WM, coregList[i], [('out_file', 'wm_seg')]),
                                     (realignList[i], applywarpList[i], [('realigned_files', 'in_file')]),
                                     (coregList[i], applywarpList[i], [('out_matrix_file', 'in_matrix_file')]),
                                     (gunzip_anat, applywarpList[i], [('out_file', 'reference')]),
                                     (applywarpList[i], smoothList[i], [('out_file', 'in_files')]),
                                     (segment, resample, [(('native_class_images', get_gm), 'in_file'),
                                                          (('native_class_images', get_gm), 'reference')]),
                                     (resample, mask_GM, [('out_file', 'in_file')]),
                                     (smoothList[i], maskFuncList[i], [('smoothed_files', 'in_file')]),
                                     (mask_GM, maskFuncList[i], [('out_file', 'mask_file')]),
                                     (maskFuncList[i], detrendList[i], [('out_file', 'in_file')]),
                                     (artList[i], datasink, [('outlier_files', 'preproc.@outlier_files'),
                                                              ('plot_files', 'preproc.@plot_files')]),
                                     (realignList[i], datasink, [('realignment_parameters', 'preproc.@par')]),
                                     (detrendList[i], datasink, [('detrended_file', 'preproc.@func')]),
                                     (detrendList[i], gunzipDetrendList[i], [('detrended_file', 'in_file')]),
                                     (gunzipDetrendList[i], normFuncList[i], [('out_file', 'apply_to_files')]),
                                     (norm_anat, normFuncList[i], [('deformation_field', 'deformation_file')]),
                                     (normFuncList[i], datasink, [('normalized_files', 'preproc.@wfunc')])
                                    ])
    else:    
        # workflow without slice timing
        for i, task in enumerate(taskList):
            preprocList.append('preproc_' + task)
            preprocList[i] = pe.Workflow(name='work_preproc_' + task, base_dir=suboDir)
            preprocList[i].connect([(sfList[i], gunzip_anat, [('anat', 'in_file')]),
                                     (sfList[i], gunzipFuncList[i], [('func', 'in_file')]),
                                     (gunzipFuncList[i], realignList[i], [('out_file', 'in_files')]),
                                     (realignList[i], artList[i], [('realigned_files', 'realigned_files'),
                                                                     ('realignment_parameters', 'realignment_parameters')]),
                                     (gunzip_anat, segment, [('out_file', 'channel_files')]),
                                     (gunzip_anat, norm_anat, [('out_file' , 'image_to_align')]),
                                     (norm_anat, datasink, [('normalized_image', 'preproc.@wanat')]),
                                     (gunzip_anat, coregList[i], [('out_file', 'reference')]),
                                     (realignList[i], coregList[i], [('mean_image', 'in_file')]),
                                     (segment, threshold_WM, [(('native_class_images', get_wm), 'in_file')]),
                                     (threshold_WM, coregList[i], [('out_file', 'wm_seg')]),
                                     (realignList[i], applywarpList[i], [('realigned_files', 'in_file')]),
                                     (coregList[i], applywarpList[i], [('out_matrix_file', 'in_matrix_file')]),
                                     (gunzip_anat, applywarpList[i], [('out_file', 'reference')]),
                                     (applywarpList[i], smoothList[i], [('out_file', 'in_files')]),
                                     (segment, resample, [(('native_class_images', get_gm), 'in_file'),
                                                          (('native_class_images', get_gm), 'reference')]),
                                     (resample, mask_GM, [('out_file', 'in_file')]),
                                     (smoothList[i], maskFuncList[i], [('smoothed_files', 'in_file')]),
                                     (mask_GM, maskFuncList[i], [('out_file', 'mask_file')]),
                                     (maskFuncList[i], detrendList[i], [('out_file', 'in_file')]),
                                     (artList[i], datasink, [('outlier_files', 'preproc.@outlier_files'),
                                                              ('plot_files', 'preproc.@plot_files')]),
                                     (realignList[i], datasink, [('realignment_parameters', 'preproc.@par')]),
                                     (detrendList[i], datasink, [('detrended_file', 'preproc.@func')]),
                                     (detrendList[i], gunzipDetrendList[i], [('detrended_file', 'in_file')]),
                                     (gunzipDetrendList[i], normFuncList[i], [('out_file', 'apply_to_files')]),
                                     (norm_anat, normFuncList[i], [('deformation_field', 'deformation_file')]),
                                     (normFuncList[i], datasink, [('normalized_files', 'preproc.@wfunc')])
                                    ])
        
    allLists = [gunzipFuncList, slicetimingParamList, sfList, slicetimeList, mcflirtList, realignList, artList,
                 coregList, applywarpList, smoothList, maskFuncList, detrendList,
                 normFuncList, gunzipDetrendList, datasink, preprocList]
    
    return allLists, outputFolder
        


# ### 1st Level functions

# In[33]:


def make1stleveltemplateslist():
    """
    Procedure to create a list of templates for 1Level analysis
    pat2Proc, taskList, subID are defined
    
    """

    firstlevelTemplatesList = [] 
    
    preprocDir = opj(oDir, 'results_' + subID + '-N', 'preproc')
    
    if 'FLAIR' in anatImg:
        anat1stLevel = opj(preprocDir, 'w' + subID + '_FLAIR.nii')
    else:
        anat1stLevel = opj(preprocDir, 'w' + subID + '_T1w.nii')

    for i, task in enumerate(taskList):
        firstlevelTemplatesList.append('templates_' + task)
        firstlevelTemplatesList[i] = {
        'anat': anat1stLevel,
        'func': opj(preprocDir, 'wdetrend-' + task + '.nii'),
        'mc_param': opj(preprocDir, 'rp_' + subID + '_task-' + task + '_bold.txt'),
        'outliers': opj(preprocDir, 'art.r' + subID + '_task-' + task + '_bold_outliers.txt'),
        'events': opj(subDir, 'func', subID + '_task-' + task + '_events.tsv')}

    # check if the files exist
    for i, item in enumerate(firstlevelTemplatesList):
        for key in firstlevelTemplatesList[i].keys():
            ffile = firstlevelTemplatesList[i][key]
            try:
                os.path.isfile(ffile)
                print("%s OK!" % ffile)
            except:
                print("%s NOT FOUND!" % ffile)
    
    return firstlevelTemplatesList


# In[34]:


def firstlevel_fMRI_UZG():
    """
    This function organizes the workflow for 1st level analysis
    subID = the definition of the subject in /data/Patients (sub-<accession number>)
    pat2Proc, taskList, firstlevelTemplatesList are defined
    
    """

    contrastList = []
    subjectInfoList = []
    sf1stlevelList = []
    modelspecList = []
    level1designList = []
    level1estimateList = []
    level1conestList = []
    analysis1stList = []
    
    # Create SelectFiles nodes
    for i, task in enumerate(taskList):
        sf1stlevelList.append('sf_' + task)
        sf1stlevelList[i] = pe.Node(interface = SelectFiles(firstlevelTemplatesList[i],
                                                              base_directory=wDir,
                                                              sort_filelist=True),
                                                              name='selectfiles_' + task)
        
    #get paradigm info
    for i, task in enumerate(taskList):
        eventFile = firstlevelTemplatesList[i]['events']
        #trialInfo = pd.read_table('/data/BIDS_source/' + task + '_seconds.tsv')
        trialInfo = pd.read_table(eventFile)

        conditions = []
        onsets = []
        durations = []

        for group in trialInfo.groupby('trial_type'):
            conditions.append(group[0])
            onsets.append(list(group[1].onset))
            durations.append(group[1].duration.tolist())

        subjectInfo = [Bunch(     
                            conditions=conditions,
                            onsets=onsets,
                            durations=durations,
                            )]

        subjectInfoList.append('subject_info_' + task)
        subjectInfoList[i] = subjectInfo

        #conditionNames = []
        #if task == 'enco_new':
        #    conditionNames = [conditions[1], conditions[0]]
        #else:
        conditionNames = [conditions[0], conditions[1]]

        # Contrasts
        cont01 = ['average', 'T', conditionNames, [1/2., 1/2.]]
        cont02 = [conditionNames[0], 'T', conditionNames, [1, 0]]
        cont03 = [conditionNames[1], 'T', conditionNames, [0, 1]]
        cont04 = [conditionNames[0] + ' >> ' + conditionNames[1], 'T', conditionNames, [1, -1]]
        cont05 = [conditionNames[1] + ' >> ' + conditionNames[0], 'T', conditionNames, [-1, 1]]

        
        contrastList.append('contrastList_' + task)
        contrastList[i] = [cont01, cont02, cont03, cont04, cont05]
        
    #define the nodes
    for i, task in enumerate(taskList):
        modelspecList.append('modelspec_' + task)
        modelspecList[i] = pe.Node(interface = SpecifySPMModel(
                                                        concatenate_runs=False,
                                                        input_units='secs',
                                                        output_units='secs',
                                                        time_repetition=slicetimingParamList[i][0],
                                                        high_pass_filter_cutoff=highPassFilterCutoffDict['enco_new'],
                                                        subject_info=subjectInfoList[i]),
                                                        name='modelspec_' + task)

        # Initiate the Level1Design nodes here
        level1designList.append('level1design_' + task)
        level1designList[i] = pe.Node(interface = spm.Level1Design(
                                                        bases={'hrf': {'derivs': [0, 0]}},
                                                        timing_units='secs',
                                                        interscan_interval=slicetimingParamList[i][0],
                                                        model_serial_correlations='AR(1)'),
                                                        name='level1design_' + task)

        # Initiate the EstimateModel nodes here
        level1estimateList.append('level1estimate_' + task)
        level1estimateList[i] = pe.Node(interface = spm
        .EstimateModel(
                                                        estimation_method={'Classical': 1}),
                                                        name='level1estimate_' + task)

        # Initiate the EstimateContrast nodes here
        level1conestList.append('level1conest_' + task)
        level1conestList[i] = pe.Node(interface = spm.EstimateContrast(
                                                        contrasts=contrastList[i]),
                                                        name='level1conest_' + task)

    #connect into workflow
    for i, task in enumerate(taskList):
        analysis1stList.append('analysis1st_' + task)
        analysis1stList[i] = pe.Workflow(name='work_1st_' + task, base_dir=opj(oDir, subID))
        analysis1stList[i].connect([
            (sf1stlevelList[i], modelspecList[i], [('func', 'functional_runs')]),
            (modelspecList[i], level1designList[i], [('session_info', 'session_info')]),
            (level1designList[i], level1estimateList[i], [('spm_mat_file', 'spm_mat_file')]),
            (level1estimateList[i], level1conestList[i],
                 [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'), ('residual_image', 'residual_image')]),
            (sf1stlevelList[i], modelspecList[i], [('mc_param', 'realignment_parameters'), ('outliers', 'outlier_files')]),
            (level1conestList[i], datasink, 
                 [('spm_mat_file', '1stLevel.@spm_mat'), ('spmT_images', '1stLevel.@T'), ('spmF_images', '1stLevel.@F'), ('con_images', '1stLevel.@con')])
                                   ])
        
    all1stlevelLists = [contrastList, subjectInfoList, sf1stlevelList, modelspecList, level1designList, level1estimateList, level1conestList, analysis1stList]
    
    return all1stlevelLists
    


# ### Update JSON files & tsv file

# In[35]:


def updatejson_preproc():
    """
    This function just updates the json log file after preprocessing
    pat2Proc, patList, subID and subLogFile are defined
    
    """
   
    preprocessed = []
    
    preprocDir = opj(oDir, 'results_' + subID + '-N', 'preproc')
     
    for task in taskList:
        ffile = opj(preprocDir, 'wdetrend-' + task + '.nii')
        if os.path.isfile(ffile):
            preprocessed.append(task)
        else: 
            print(('%s not found' % ffile))
    
    with open(subLogFile, "r") as jFile: jData = json.load(jFile)       
    jData["fMRI_preproc"] = preprocessed
    
    with open(subLogFile, "w") as jFile: json.dump(jData, jFile)
    #print(json.dumps(jData, indent=4, sort_keys=True))


# In[36]:


def updatejson_1stlevel():
    """
    This function updates the json logfile after 1st level processing
    pat2Proc, patList, subID and subLogFile are defined

    """
    
    firstlevelList = []
    
    firstlevelDir = opj(oDir, 'results_' + subID + '-N', '1stLevel')

    for task in taskList:
        spmFile = opj(firstlevelDir, task, 'SPM.mat')
        if os.path.isfile(spmFile):
            firstlevelList.append(task)
        else: 
            print(('%s not found' % spmFile))

    with open(subLogFile, "r") as jFile: jData = json.load(jFile)
    jData["Statistics"] = firstlevelList

    with open(subLogFile, "w") as jFile: json.dump(jData, jFile)
    print(json.dumps(jData, indent=4, sort_keys=True))


# In[37]:


def update2tsv():
    """add a line in the subjects.tsv file in wDir"""
    
    line2Add = []
    
    # define the variables
    subName = pat2Proc['Patient Name'] 
    subAdrema = pat2Proc['Patient ID']
    subAccession = pat2Proc['Accession Number']
    subDate= pat2Proc['Study Date']
    
    #with open(opj(iDir, 'subjects.tsv'), 'r') as tsvFile:
    #    reader = csv.reader(tsvFile, delimiter='\t')
    #    # read the number of rows, this is the next row number because of headers
    #    rowCount = sum(1 for row in reader)
    
    with open(subLogFile) as jFile:
        jData = json.load(jFile)
        subComment = jData['Referral']
        
    if len(os.listdir(opj(wDir, subID, 'dwi'))) == 0:
        dwiValue = 0
    else:
        dwiValue = 1
           
    #if len(os.listdir(opj(wDir, subID, 'asl'))) == 0:
    #    aslValue = 0
    #else:
    #    aslValue = 1
        
    if len(os.listdir(opj(wDir, subID, 'fmap'))) == 0:
        fmapValue = 0
    else:
        fmapValue = 1
    
    line2Add.append(subName)
    line2Add.append(subAdrema)
    line2Add.append(subAccession)
    line2Add.append(subDate)
    line2Add.append(subComment)
    line2Add.append(str(taskList))
    line2Add.append(str(dwiValue))
    line2Add.append(str(fmapValue))
    line2Add.append('N') # for the time being no figs
    line2Add.append('N') # for the time being no radiological report
    
    return line2Add


# In[38]:


def all_patients():
    """
    function cheques all .json files in /data/Patients/SubjectInfoFiles
    
    """
    
    for idx, subject in enumerate([x for x in os.listdir(opj(wDir, 'SubjectInfoFiles')) if x.endswith('.json')]):
        with open(opj(wDir, 'SubjectInfoFiles', subject), 'r') as jFile:
            jData = json.load(jFile)
        print('%s -> %s' % (subject, jData['Subject DCM tags']['Patient Name']))


# In[39]:


def show_subjects():
    """
    function to display a tabel of the subjects in /data/Patients/SubjectInfoFiles/subjects.tsv
    
    """

    print(' #   Subject                                  Adrema          Accession  Date Exam  Referral                                           fMRI tasks                                                       DTI     F-map     Figs    Report')
    print('-'*220)
    with open(opj(iDir, 'subjects.tsv'), 'rt') as tsvFile:
        reader = csv.reader(tsvFile, delimiter='\t')
        #sortedLines = sorted(reader, key=operator.itemgetter(3))
        idx = 0
        for row in reader:
            idx = idx + 1
            row = [idx] + row
            print(' {:<3} {:<40} {:<15} {:<10} {:<10} {:<50} {:<65} {:<8} {:<8} {:<8} {:<8}'.format(*row))


# In[40]:


def update_subjects():
    """
    Function to update certain parameters in /data/Patients/SubjectInfoFiles/subjects.tsv
    
    """
    
    show_subjects()
    
    with open(opj(wDir, 'SubjectInfoFiles', 'subjects.tsv'), 'rt') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        lines = list(reader)

    rows = len(lines) 
    
    try:
        answer = int(input('\nWelke lijn wilt u aanpassen?:'))-1
        print(lines[answer])
        
        param2Change = {'Referral':4, 'Figs':8, 'Report':9}

        if answer in range(rows) and answer != 0:
            
            item = input('\nWelke waarde wil je veranderen? (kies uit: %s): ' % (', '.join(str(i) for i in list(param2Change.keys()))))
            if item in param2Change.keys():
                position = param2Change[item]
                print("\nHuidige waarde variable '%s' = %s" % (item, lines[answer][position]))
                newvalue = input("\nNieuwe waarde voor variabele '%s' = " % item)
                lines[answer][position] = newvalue
                
                print(lines[answer])
                
                with open(opj(wDir, 'SubjectInfoFiles', 'newsubjects.tsv'), 'w') as tsv_file:
                    writer = csv.writer(tsv_file, delimiter='\t')
                    writer.writerows(lines)
                
                shutil.copy(opj(wDir, 'SubjectInfoFiles', 'subjects.tsv'), opj(wDir, 'SubjectInfoFiles', 'subjects.tsv.backup'))
                shutil.copy(opj(wDir, 'SubjectInfoFiles', 'newsubjects.tsv'), opj(wDir, 'SubjectInfoFiles', 'subjects.tsv'))
                os.remove(opj(wDir, 'SubjectInfoFiles', 'newsubjects.tsv'))
                
                clear_output()
                
                print('Dit is the aangepaste subjects.tsv file:\n')
                show_subjects()
                
            else:
                print('Deze parameter bestaat niet!')

        else:
            print('Deze rij bestaat niet (of is hoofding)!')

    except:
        print('Fout! Duid een lijn aan met een nummer!')


# # Start Procedure!

# In[41]:


starttime = time.time()


# In[42]:


notProcessedList, pat2Proc, subID, subDir = select_patient_fMRI('name')


# In[43]:


# Create the BIDS dir and subdirs
mkbidsdir(pat2Proc)


# In[44]:


# Create the json log file
subLogFile = create_json_log_file()


# In[85]:


pat2Proc


# In[46]:


#subID = 'sub-' + pat2Proc['Accession Number']
#subDir = opj(wDir, subID)
#subLogFile = opj(wDir, 'SubjectInfoFiles', subID + '_log.json')


# ## Dicom to NIFTI

# In[47]:


# convert dicom images to niftii
dcm2niftii()


# In[48]:


sDict = f_series_numbers(subDir)


# In[49]:


pDict = f_series_timepoints(subDir)


# In[50]:


sDict


# In[51]:


pDict


# In[52]:


# change copy2dirs to still include non-completed runs and to introduce run-1, run-2, etc if paradiagms where executed more than once
# Keep the info of how many volumes each run contains in a separate file
# Also copy the ASL en M0 files to subDir/asl


# In[53]:


newcopy2dirs() # IF ANY FILES HAVE TO RENAMED, DO IT BEFORE RUNNING THIS!


# In[54]:


show_asl()


# In[55]:


taskList = cre_task_list()


# In[56]:


taskList #= ['read', 'wgen']


# ## DWI, fmap and rename

# In[57]:


# get rid of unwanted dwi files
dwi2waste()


# In[58]:


rename_anat()


# In[59]:


# reprogram the rename_func() function to take into acount different runs and in the events file the possibility to truncate to any number of scans below the number of actual scans (interactive)


# In[60]:


rename_func()


# In[61]:


rename_fmap()


# In[62]:


rename_dwi()


# In[63]:


rename_asl()


# In[64]:


templatesList, anatImg = maketemplateslist()


# In[65]:


templatesList


# ## Preprocessing

# In[66]:


allLists, outputFolder = preproc_fMRI_UZG(ST= 'Y')


# In[67]:


gunzipFuncList = allLists[0]
slicetimingParamList = allLists[1]
sfList = allLists[2]
slicetimeList = allLists[3]
mcflirtList = allLists[4]
realignList = allLists[5]
artList = allLists[6]
coregList = allLists[7]
applywarpList = allLists[8]
smoothList = allLists[9]
maskFuncList = allLists[10]
detrendList = allLists[11]
normFuncList = allLists[12]
gunzipDetrendList = allLists[13]
datasink = allLists[14]
preprocList = allLists[15]


# In[68]:


## Use the following substitutions for the DataSink output
substitutions = [('asub', 'sub'),
                 ('_roi_mcf', ''),
                 ('.nii.gz.par', '.par'),
                 ]

datasink.inputs.substitutions = substitutions


# In[69]:


# run the processing
# all preprocessed data are in the /data/output/output_folder/preproc dir

timePreprocDict = {}

for i, task in enumerate(taskList):
    #task = re.sub('_', '', task)
    start = time.time()
    print('Start processing %s' % task.upper())
    preprocList[i].run('MultiProc');
    end = time.time()
    temp = end-start
    timePreprocDict[task] = temp

    # rename the detrend and wdetrend file in the datasink dir
    os.chdir(opj(oDir, outputFolder, 'preproc'))
    os.rename('detrend.nii.gz', 'detrend-' + task + '.nii.gz')
    os.rename('wdetrend.nii', 'wdetrend-' + task + '.nii')
    
updatejson_preproc()


# In[70]:


# Create preproc output graphs
for i, task in enumerate(taskList):
    preprocList[i].write_graph(graph2use='colored', format='png', simple_form=True)


# In[71]:


updatejson_preproc()


# In[72]:


for key in timePreprocDict:
    procTime = timePreprocDict[key]
    hours = procTime//3600
    proc_time = procTime - 3600*hours
    minutes = procTime//60
    seconds = procTime - 60*minutes
    print('Total preproc time for ' + key + ' = %d:%d:%d' %(hours,minutes,seconds))


# ## Level1

# In[73]:


firstlevelTemplatesList = make1stleveltemplateslist()


# In[74]:


all1stlevelLists = firstlevel_fMRI_UZG()


# In[75]:


contrastList = all1stlevelLists[0]
subjectInfo_List = all1stlevelLists[1]
sf1stlevelList = all1stlevelLists[2]
modelspecList = all1stlevelLists[3]
level1designList = all1stlevelLists[4]
level1estimateList = all1stlevelLists[5]
level1conestList = all1stlevelLists[6]
analysis1stList = all1stlevelLists[7]


# In[76]:


# perform actual 1st level analysis
time1levelDict = {}

for i, task in enumerate(taskList):
    
    start = time.time()
    analysis1stList[i].run()
    end = time.time()
    temp = end-start
    time1levelDict[task] = temp
    
    #copy the SPM & wcon & wess files to the relevant results directory
    
    SPMFilesDir = opj(oDir, outputFolder, '1stLevel')
    SPMFiles = os.listdir(SPMFilesDir)
    
    # create task directories in the datasink 1stLevel dir
    if not os.path.exists(opj(SPMFilesDir, task)):
        os.makedirs(opj(SPMFilesDir, task))
        
    for f in SPMFiles:
        fullF = opj(SPMFilesDir, f)
        if (os.path.isfile(fullF)):
                try:
                    shutil.move(fullF, opj(SPMFilesDir, task))
                except:
                    print('Unable to copy file %s.' % fullF)


# In[77]:


# Create preproc output graphs
for i, task in enumerate(taskList):
    analysis1stList[i].write_graph(graph2use='colored', format='png', simple_form=True)


# In[78]:


for key in time1levelDict:
    procTime = time1levelDict[key]
    hours = procTime//3600
    procTime = procTime - 3600*hours
    minutes = procTime//60
    seconds = procTime - 60*minutes
    print('Total 1st level process time for ' + key + ' = %02d:%02d:%02d' %(hours,minutes,seconds))


# In[79]:


updatejson_1stlevel()


# In[80]:


stoptime = time.time()


# In[81]:


totalprocTime = stoptime-starttime
hours = totalprocTime//3600
minutesprocTime = totalprocTime - 3600*hours
minutes = minutesprocTime//60
seconds = minutesprocTime - 60*minutes
print('Total processing time = %02d:%02d:%02d' %(hours,minutes,seconds))


# # Write info to subject.tsv in wdir

# In[82]:


line2Add = update2tsv()


# In[83]:


with open(opj(iDir, 'subjects.tsv'), 'at') as tsvFile:
    tsvFile.write('\t'.join(line2Add) + '\n')


# In[84]:


print(' #   Subject                                  Adrema          Accession  Date Exam  Referral                                 fMRI tasks                                                       DTI     F-map     Figs    Report')
print('_'*223)
print()
with open(opj(iDir, 'subjects.tsv'), 'rt') as tsvFile:
    reader = csv.reader(tsvFile, delimiter='\t')
    sortedLines = sorted(reader, key=operator.itemgetter(3))
    idx = 0
    for row in sortedLines:
        idx = idx + 1
        row = [idx] + row
        print(' {:<3} {:<40} {:<15} {:<10} {:<10} {:<50} {:<65} {:<8} {:<8} {:<8} {:<8}'.format(*row))


# In[ ]:




