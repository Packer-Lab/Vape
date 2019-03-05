import numpy as np
import tifffile as tf
import ntpath
import os



def get_tiffs(path):
    
    tiff_files = []
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.tiff'):
            tiff_files.append(os.path.join(path,file))
                   
    return tiff_files

 
def interpolate_through_stim(stack=None, db=None, threshold=1.5):
    
    '''  
    remove frames with mean pixel intensity higher than 
    threhold x average across time series
    returns stack with linear interpolation through these stim frames 
    
    takes input of stack OR db
    
    if stack: returns artifact cleaned stack array
    if db: returns db dict with path now to cleaned tiff
         
    '''

    if not stack and not db: raise ValueError('must pass function stack or db')
    if stack and db: raise ValueError('do not pass function stack and db')
       
    if db: 
        tiff_files = get_tiffs(db['data_path'][0])
        
        if len(tiff_files) > 1: raise ValueError('can only support single tiff file in folder currently')
        tiff_file = tiff_files[0]
        stack = tf.imread(tiff_file)

    dims = stack.shape
    n_frames = dims[0]
    
    av_frame = np.mean(stack, axis=(1,2))
    
    # Frames with averge fluoresence threshold * higher than average
    to_remove = np.where(av_frame > threshold*np.mean(av_frame))[0]

    #remove the frames that are above threshold
    blanked = np.delete(stack, to_remove, axis=0)

    # list of frames not blanked
    xt = np.arange(n_frames)
    xt = np.delete(xt,to_remove)

    #perform pixel wise linear interpolation across blanked frames
    for row in range(dims[1]):
        for col in range(dims[2]):

            px = blanked[:,row,col]

            intp = np.interp(to_remove, xt, px)

            stack[to_remove,row,col] = intp
            
    assert stack.shape == dims

    
    if db:
    
        #update db path
        ar_path = os.path.join(os.path.dirname(tiff_file), 'artifactRemoved')            
        db['data_path'] = [ar_path]   
        if not os.path.exists(ar_path): os.makedirs(ar_path)
         
        #write artifact removed stack
        exp_name = ntpath.basename(tiff_file).split('.')[0]
        output_path = os.path.join(ar_path, exp_name +'_artifactRemoved.tiff')
        tf.imwrite(output_path, stack, photometric='minisblack')
       
        return db
    
    else:
        return stack






















