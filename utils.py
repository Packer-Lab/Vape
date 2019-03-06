import numpy as np
import tifffile as tf
import ntpath
import os
from lxml import objectify
from lxml import etree



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
         
    (JR 2019) 
     
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

        
        
class ParseMarkpoints():  
    
    '''
    parses gpls or xmls from markpoints into python variables
    copied from Blimp (JR 2019)
    
    '''

    def __init__(self, xml_path=None, gpl_path=None):
        
        #super(ParseMarkpoints, self).__init__()
        
        self.xml_path = xml_path
        self.gpl_path = gpl_path
        
        if self.gpl_path:            
            self.parse_gpl()
            
        if self.xml_path:
            self.parse_xml()

       
        
    def parse_gpl(self):
        
        '''extract information from naparm gpl file'''
        
        gpl = etree.parse(self.gpl_path)

        self.Xs = []
        self.Ys = []
        self.is_spirals = []
        self.spiral_sizes = []

        for elem in gpl.iter():

            if elem.tag == 'PVGalvoPoint':
                self.Xs.append(elem.attrib['X'])
                self.Ys.append(elem.attrib['Y'])
                self.is_spirals.append(elem.attrib['IsSpiral'])
                #the spiral size gpl attrib contains a positive and negative number for 
                #some reason. This takes just the positive, may cause errors later
                self.spiral_sizes.append(elem.attrib['SpiralSize'].split(' ')[0])
                
            
    def parse_xml(self):

        '''extract requried information from xml into python lists'''

        xml = etree.parse(self.xml_path)

        self.laser_powers = []
        self.spiral_revolutions = []
        self.durations = []
        self.initial_delays = []
        self.repetitions = []
        
        for elem in xml.iter():

            if elem.tag == 'PVSavedMarkPointSeriesElements':
                self.iterations = elem.attrib['Iterations']
            
            if elem.tag == 'PVMarkPointElement':
                self.laser_powers.append(elem.attrib['UncagingLaserPower'])
                self.repetitions.append(elem.attrib['Repetitions'])
                
            elif elem.tag == 'PVGalvoPointElement':
                self.durations.append(elem.attrib['Duration'])
                self.spiral_revolutions.append(elem.attrib['SpiralRevolutions'])
                self.initial_delays.append(elem.attrib['InitialDelay'])
                
               
        #detect if a dummy point has been sent and remove it if so
        if self.laser_powers[0] == '0':
            self.dummy = True
            self.laser_powers.pop(0)
            self.spiral_revolutions.pop(0)
            self.durations.pop(0)
            self.initial_delays.pop(0)
            self.repetitions.pop(0)




















