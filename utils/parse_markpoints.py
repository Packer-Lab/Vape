from lxml import objectify
from lxml import etree


class ParseMarkpoints():  
    
    ''' parses gpls or xmls from markpoints into python variables'''

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

            elif elem.tag == 'PVMarkPointElement':
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
        
        
    def build_strings(self, **kwargs):
    
        '''builds string using variables from parsed gpls and xmls that can be sent via prairie link'''

        for k,v in kwargs.items():
            setattr(self,k,v)
            
        markpoints_strings = []
        
        settings_string = '{0} {1} {2} {3} {4} {5} {6} {7} 0.12 '.format(self.X,self.Y,self.duration, \
                                                                        'Uncaging', self.laser_power, self.is_spiral, \
                                                                        self.spiral_size, self.spiral_revolutions)    
        #repeat num spirals times
        markpoints_string = settings_string * int(self.num_spirals)
        
        #snip the inter-spiral-delay off the last point and add the markpoints command at the start
        markpoints_string  = '-mp ' + markpoints_string[:-5]
        #Rmarkpoints_string = markpoints_string + ' \n'
                   
        return markpoints_string
        
        
    def groups_strings(self,inter_group_interval, group_list, SLM_trigger=False, n_repeats=1):
    
        '''
        takes an input of n_groups length markpoints_string list and concatenates to one markpoints command string
        groups are stimmed with an interval os inter_group_interval (ms)
        
        Set SLM_trigger to True to add a trigger "laser" string in between stimulation strings. This sends a 5V to the SLM to trigger
        a phase mask change.     
                  
        '''
        
        n_groups = len(group_list)
        
        assert n_groups > 1, 'must give at least 2 markpoints strings to concantenate'
     
        all_groups = ''
        
        for idx, group in enumerate(group_list):
                     
            #pop off the -mp from all groups following group 1
            if idx != 0:
                group = group[4:]
            
            all_groups += group
            
            # add the inter group interval if it is not the final group
            if idx != n_groups-1 and not SLM_trigger:
                            
                all_groups += ' ' + str(inter_group_interval) + ' '     
            
            elif idx != n_groups-1 and SLM_trigger:     
            
                #the x and y values of the group just stimmed
                x = (group.split(' ')[1] if idx == 0 else group.split(' ')[0])
                y = (group.split(' ')[2] if idx == 0 else group.split(' ')[1])
                
                trigger_string = self.misc_stims(x, y, 'trigger', inter_group_interval=inter_group_interval)
                all_groups += trigger_string
                
        # repeat this string if required (uses trigger galvo position of last group in repeat
        if n_repeats > 1:
            repeat_string = trigger_string + all_groups[4:]
            all_groups = all_groups + (repeat_string * n_repeats) 
          
        #get rid of random space on end
        all_groups = all_groups[:-1]
          
        return all_groups

    def misc_stims(self, x, y, stim_type, inter_group_interval=None):
    
        '''
           used to build an intial dummy string (set stim_type to dummy) or a trigger "laser" pulse           
           (set stim_type to trigger)
           
           currently uses a 1ms pulse duration for triggering with a 0.12ms delay
           add the inter_group_interval argument if using triggers 
           
        '''
        
        if stim_type == 'dummy':       
            return '-MarkPoints {0} {1} {2} {3} {4}'.format(x, y, '1', 'Uncaging', '0')
        
        elif stim_type == 'trigger':
        
            trigger_len = 5 #ms
            
            #how long to delay between trigger and next spiral
            delay = inter_group_interval - trigger_len - 0.12   
                    
            return '{0} {1} {2} {3} {4} {5} {6} '.format(0.12, x, y, trigger_len, 'Trigger', 1000, delay)
            
        else:
            raise ValueError('misc stimulus type not recognised')
        
            
            
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
                    
    
    
    
    
    
    
    
    
    
    
    
    
    


