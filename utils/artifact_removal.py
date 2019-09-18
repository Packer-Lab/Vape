import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tifffile import *
import numpy as np
from skimage.measure import label, regionprops, find_contours
import time
#import cv2

from scipy.signal import convolve2d


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

#FUTURE?: instead of removing pixel information, replace with average of surround
# def eight_neighbor_average_convolve2d(x):
#     kernel = np.ones((3, 3))
#     kernel[1, 1] = 0

#     neighbor_sum = convolve2d(
#         x, kernel, mode='same',
#         boundary='fill', fillvalue=0)

#     num_neighbor = convolve2d(
#         np.ones(x.shape), kernel, mode='same',
#         boundary='fill', fillvalue=0)

#     return neighbor_sum / num_neighbor

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
        if width < width_thresh:
            
            frame[rows, cols] = 0

            #FUTURE?: instead of removing pixel information, replace with average of surround
#             neighbour_avg = eight_neighbor_average_convolve2d(frame)
            
#             frame[rows,cols] = neighbour_avg[rows, cols]
            
            #useful for debugging
            #labelled[rows, cols] = 0
        else:
            pass
            #labelled[rows, cols] = width

    return frame


def artifact_removal(stack, thresh_list='up_to_stim1', remove_me='all', sigma=2, width_thresh=10, nplanes=1):

    '''
    main function for photostimulation artifact removal
    --------
    inputs
    --------
    stack: iamge stack on which to perform removal
    thresh_list: list of frames to average to get baseline pixel values (so frames without stim artifact)
                 defaults to all frames up to the first remove_me frame
    remove me: list of frames to run artifact removal algorithm (defaults to all frames in stack)
    sigma: sigma value of pixel intensity disribution above to mark as potentially contaminated
    width_thresh: groups of connected pixels with width less than this value will be removed
    --------
    returns
    stack: stack with artifact removed
    '''

    if thresh_list == 'up_to_stim1' and remove_me == 'all':
        raise ValueError('cannot find stim1 if removing all frames')

    if remove_me == 'all':
        remove_me = range(stack.shape[0])

    if thresh_list == 'up_to_stim1':
        thresh_list = range(remove_me[0]-1)
    
    if nplanes > 1:
#         thresh = np.empty((nplanes,stack.shape[1],stack.shape[2]))
        
        for i in range(nplanes):
            frame_list = range(i,stack.shape[0],nplanes)
            
            thresh_list_sliced = [t for t in thresh_list if t in frame_list]
            
            thresh = find_threshold(stack[i::nplanes], thresh_list_sliced, sigma=sigma)
            
            for frame_idx in remove_me:
                if frame_idx in frame_list:
                    frame = stack[frame_idx, :, :]

                    frame_bin = binarise_frame(frame, thresh)

                    processed_frame = process_frame(frame, frame_bin, width_thresh)

                    stack[frame_idx, :, :] = processed_frame

        return stack
    
    else:
        
        thresh = find_threshold(stack, thresh_list, sigma=sigma)

        for frame_idx in remove_me:

            frame = stack[frame_idx, :, :]

            frame_bin = binarise_frame(frame, thresh)

            processed_frame = process_frame(frame, frame_bin, width_thresh)

            stack[frame_idx, :, :] = processed_frame

        return stack
