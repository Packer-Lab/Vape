from tifffile import *
import numpy as np
from skimage.measure import label, regionprops, find_contours
import time
import cv2

def find_threshold(tiff, thresh_list, sigma):
    
    '''
    tiff: tiff stack 
    stim_start: frame on which the stimulus started
    sigma: sigma value over which to threshold pixels

    thresh: thresholded single frame
    '''
         
    base_vals = tiff[thresh_list, :, :]
    base_mean = np.mean(base_vals, 0)
    base_std =  np.std(base_vals, 0)
    thresh = base_mean + base_std*sigma
    
    return thresh 
    
    
    
def binarise_frame(frame, thresh):
    
    '''binarises frame, where all pixels > thresh = 1'''
    
    assert frame.shape == thresh.shape
    
    return np.greater(frame, thresh).astype('int')



def process_frame(frame, frame_bin, width_thresh=10):
    
    '''
    finds regions of connected pixels in a frame and their widths
    returns:
    labelled: array of frame dimensions with each pixels region labelled
    widths: the width (x len) of each region
    '''
    #find regions of pixel connectivity
    labelled, num_labels = label(frame_bin, connectivity=1, return_num=True)
    
    # the properties of connected regions
    regions = regionprops(labelled)

    for i,props in enumerate(regions):
        
        #if i==0:continue
        
        coords = props['Coordinates']
        
        rows = coords[:,0]
        cols = coords[:,1]
        
        width_rows = max(rows) - min(rows)
        width_cols = max(cols) - min(cols)
        
        width = (width_rows if width_rows<width_cols else width_cols)
        
        # the width of the labelled region is thin or it is very asymmetrical
        if width < width_thresh or props['major_axis_length'] / props['minor_axis_length'] > 2:
            
            frame[rows, cols] = 0
            
            #useful for debugging
            #labelled[rows, cols] = 0
        else:
            pass
            #labelled[rows, cols] = width
  
    return frame   



def artifact_removal(tiff_path, thresh_list, remove_me='all', sigma=2, width_thresh=10):

    '''
    main function for photostimulation artifact removal
    --------
    inputs
    --------
    tiff_path: path to tiff to perform removal
    thresh_list: list of frames to average to get baseline pixel values (so frames without stim artifact)
    remove me: list of frames to run artifact removal algorithm (defaults to all frames in tiff)
    sigma: sigma value of pixel intensity disribution above to mark as potentially contaminated
    width_thresh: groups of connected pixels with width greater than this value will be removed
    --------
    returns
    tiff: tiff with artifact removed
    '''

    
    tiff = imread(tiff_path)
    
    thresh = find_threshold(tiff, thresh_list, sigma=sigma)
    
    for frame_idx in remove_me:

        frame = tiff[frame_idx, :, :]

        frame_bin = binarise_frame(frame, thresh)
        
        processed_frame = process_frame(frame, frame_bin, width_thresh)

        tiff[frame_idx, :, :] = processed_frame

    return tiff
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
