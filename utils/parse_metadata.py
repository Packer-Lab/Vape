from utils.gsheets_importer import path_finder

def getPVStateShard(path, key):
    
    import xml.etree.ElementTree as ET
    
    value = []
    description = []
    index = []
    
    xml_tree = ET.parse(path) # parse xml from a path
    root = xml_tree.getroot() # make xml tree structure

    pv_state_shard = root.find('PVStateShard') # find pv state shard element in root

    for elem in pv_state_shard: # for each element in pv state shard, find the value for the specified key
        
        if elem.get('key') == key: 
            
            if len(elem) == 0: # if the element has only one subelement
                value = elem.get('value')
                break
                
            else: # if the element has many subelements (i.e. lots of entries for that key)
                for subelem in elem:
                    value.append(subelem.get('value'))
                    description.append(subelem.get('description'))
                    index.append(subelem.get('index'))
        else:
            for subelem in elem: # if key not in element, try subelements
                if subelem.get('key') == key:
                    value = elem.get('value')
                    break
                    
        if value: # if found key in subelement, break the loop
            break
            
    if not value: # if no value found at all, raise exception
        raise Exception('ERROR: no element or subelement with that key')
    
    return value, description, index


def parsePVxml(path):
    
    import xml.etree.ElementTree as ET
    
    xml_tree = ET.parse(path) # parse xml from a path
    root = xml_tree.getroot() # make xml tree structure

    sequence = root.find('Sequence')
    acq_type = sequence.get('type')

    if 'ZSeries' in acq_type:
        n_planes = len(sequence.findall('Frame'))
        n_sequences = len(root.findall('Sequence'))
        n_frames = n_sequences * n_planes

    else:
        n_planes = 1
        n_frames = len(sequence.findall('Frame'))
    print('Number of frames:', n_frames, '\nNumber of planes:', n_planes)
    
    frame_period = float(getPVStateShard(path,'framePeriod')[0])
    fps = 1/frame_period
    print('Frames per second:', fps)

    frame_avg = int(getPVStateShard(path,'rastersPerFrame')[0])
    print('Frame averaging:', frame_avg)

    pixels_per_line = int(getPVStateShard(path,'pixelsPerLine')[0])
    print('Size (x):', pixels_per_line)

    lines_per_frame = int(getPVStateShard(path,'linesPerFrame')[0])
    print('Size (y):', lines_per_frame)

    laser_powers, lasers, _ = getPVStateShard(path,'laserPower')
    for power,laser in zip(laser_powers,lasers):
        if laser == 'Imaging':
            imaging_power = float(power)
    print('Imaging laser power:', imaging_power)
    
    pixelSize, _, index = getPVStateShard(path,'micronsPerPixel')
    for pixelSize,index in zip(pixelSize,index):
        if index == 'XAxis':
            pixelSizeX = float(pixelSize)
        if index == 'YAxis':
            pixelSizeY = float(pixelSize)
    print('Pixel size (x,y):', pixelSizeX, pixelSizeY)
    
    return fps, frame_avg, pixels_per_line, lines_per_frame, imaging_power, n_planes, n_frames, pixelSizeX, pixelSizeY


def parseNAPARMxml(path):
    
    import xml.etree.ElementTree as ET  

    xml_tree = ET.parse(path)
    root = xml_tree.getroot()

    title = root.get('Name')
    n_trials = int(root.get('Iterations'))

    for elem in root:
        if int(elem[0].get('InitialDelay')) > 0:
            inter_point_delay = int(elem[0].get('InitialDelay'))

    import re 

    n_groups, n_reps, n_shots = [int(s) for s in re.findall(r'\d+', title)]

    print('Number of groups:', n_groups, '\nNumber of sequence reps:', n_reps, '\nNumber of shots:', n_shots, '\nNumbers of trials:', n_trials, '\nInter-point delay:', inter_point_delay)
    
    repetitions = int(root[1].get('Repetitions'))
    print('Repetitions:', repetitions)
    
    return n_groups, n_reps, n_shots, n_trials, inter_point_delay, repetitions


def parseNAPARMgpl(path):
    
    import xml.etree.ElementTree as ET
    
    xml_tree = ET.parse(path)
    root = xml_tree.getroot()

    for elem in root:
        if elem.get('Duration'):
            single_stim_dur = float(elem.get('Duration'))
            print('Single stim dur (ms):', elem.get('Duration'))
            break

    return single_stim_dur


def getMetadata(tiffs_pstation, naparm_pstation):
    
    import os 
    
    pv_values = []
    naparm_xml = []
    naparm_gpl = []

    for tiff_path, naparm_path in zip(tiffs_pstation, naparm_pstation):
        print(tiff_path)
        
        PV_xml_path = []
        
        for dirname, dirs, files in os.walk(tiff_path):
            for file in files:
                if file.endswith('.xml'):
                    PV_xml_path = os.path.join(tiff_path, file)

        if not PV_xml_path:

            tiff_path_up = os.path.dirname(tiff_path)
            
            for dirname, dirs, files in os.walk(tiff_path_up):
                for file in files:
                    if file.endswith('.xml'):
                        PV_xml_path = os.path.join(tiff_path_up, file)

        print(PV_xml_path)

        NAPARM_xml_path = path_finder(naparm_path, '.xml')[0]
        print(NAPARM_xml_path)
        NAPARM_gpl_path = path_finder(naparm_path, '.gpl')[0]
        print(NAPARM_gpl_path)
        
        if(tiff_path):
            pv_values.append(parsePVxml(PV_xml_path))
        if(naparm_path):
            naparm_xml.append(parseNAPARMxml(NAPARM_xml_path))
            naparm_gpl.append(parseNAPARMgpl(NAPARM_gpl_path))

    return pv_values, naparm_xml, naparm_gpl