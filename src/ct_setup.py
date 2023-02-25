# %% ***************************************************** Import Packages *****************************************************
import numpy as np
import matplotlib.pyplot as plt
import astra 
import tigre
import GPUtil
from sys import exit
import time

# %% ********************************************************** TIGRE **********************************************************

class ct_tigre:
    '''Setup for TIGRE toolbox'''
    def __init__(self, N, N_ang, N_det, angles, fp_model, bp_model, proj_geom, source_origin, source_det, det_width):
        '''Class setup for a 2D X-ray CT problem in TIGRE  

        ----- INPUT -----
        N:              Number of pixels in an N x N image
        N_ang:          Number of angles
        N_det:          Number of detector elements
        angles:         Angles in radians
        fp_model:       Projection model, choose between 'Siddon' (line) and 'interpolated' (Joseph)
        bp_model:       Projection model, choose between 'matched' and 'FDK'
        proj_geom:      Projection geometry 'parallel' or 'cone'
        source_origin:  Distance to source origin in mm
        source_det:     Distance to source detecor in mm
        det_width:      Detector width
        '''
        # Set fields
        self.N = N
        self.N_ang = N_ang
        self.N_det = N_det
        self.proj_angles = angles
        self.m = N_ang*N_det
        self.n = N**2
        self.proj_geom = proj_geom
        self.det_width = det_width
        self.fp_model = fp_model
        self.bp_model = bp_model

        # Check if there is a GPU connected to the host
        n_device, = np.shape(GPUtil.getAvailable())

        if n_device < 1:
            raise Exception("No GPUs available.")
        
        # Define Geometry
        self.geo  = tigre.geometry()

        # VARIABLES
        # Distance 
        self.geo.DSD = source_det 
        self.geo.DSO = source_origin

        # Detector parameters
        self.geo.nDetector = np.array([1, self.N_det])                      # number of pixels
        self.geo.dDetector = np.array([self.geo.dVoxel[0], self.det_width]) # size of each pixel in mm
        self.geo.sDetector = self.geo.nDetector * self.geo.dDetector        # total size of the detector in mm

        # Image parameters 
        self.geo.nVoxel = np.array([1, self.N, self.N])     # number of voxels
        self.geo.sVoxel = np.array([1, self.N, self.N])     # total size of the image in mm
        self.geo.dVoxel = self.geo.sVoxel / self.geo.nVoxel # size of each voxel in mm

        # Offset
        self.geo.offOrigin = np.array([0, 0, 0])   # offset of image from origin in mm
        self.geo.offDetector = np.array([0, 0])    # offset of detector in mm

        # Auxiliary 
        self.geo.accuracy = 0.5 # variable to define accuracy of 'interpolated' projection, 
                                # it defines the amount of samples per voxel (recommended <= 0.5)

        self.geo.COR = 0                           # y direction displacement for centre of rotation correction in mm
        self.geo.rotDetector = np.array([0, 0, 0]) # rotation of the detector, by X, Y, and Z axis respectively in radians

        # Setup projection geometry
        if proj_geom == 'parallel' or proj_geom == 'cone': 
            self.geo.mode = proj_geom 
        else: 
            print("Projection geometry can only be parallel or cone.")

# %% ********************************************************** ASTRA **********************************************************

class ct_astra:
    '''Setup for ASTRA toolbox'''
    def __init__(self, N, N_ang, N_det, angles, proj_model, proj_geom, source_origin, origin_det, det_width, GPU = True):
        '''Class setup for a 2D X-ray CT problem in ASTRA

        ----- INPUT -----
        N:              Number of pixels in an N x N image
        N_ang:          Number of angles
        N_det:          Number of detector elements
        angles:         Angles in radians
        proj_model:     Projection model for CPU version choose between 'line', 'strip', or 'linear' (Joseph) 
                            and for GPU we only have Joseph
        proj_geom:      Projection geometry 'parallel' or 'fanflat'
        GPU:            True (use GPU) or False (use CPU)
        source_origin:  Distance between the source and the center of rotation
        origin_det:     Distance between the center of rotation and the detector array
        det_width:      Detector width
        '''
        # Set fields
        self.N = N
        self.N_ang = N_ang
        self.N_det = N_det
        self.proj_angles = angles
        self.m = N_ang*N_det
        self.n = N**2
        self.vol_geom  = astra.create_vol_geom(self.N,self.N)
        self.GPU = GPU

        # Check if there is a GPU connected to the host
        if GPU == True:
            n_device, = np.shape(GPUtil.getAvailable())

            if n_device < 1:
                raise Exception("No GPUs available.")

        # SETUP PROJECTION GEOMETRY
        # Parallel beam geometry
        if proj_geom == 'parallel': 
            self.proj_geom = astra.create_proj_geom(proj_geom, det_width, N_det, self.proj_angles)
            if GPU == True:
                self.proj_id   = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            else:
                self.proj_id   = astra.create_projector(proj_model, self.proj_geom, self.vol_geom)

        # Fan beam geometry
        elif proj_geom == 'fanflat':
            if proj_model == 'linear' and GPU == False:
                raise Exception("Fan beam geometry using the CPU can only handle strip and line")

            self.proj_geom = astra.create_proj_geom(proj_geom,det_width,N_det,self.proj_angles,source_origin,origin_det)
            if GPU == True:
                self.proj_id   = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            else:
                self.proj_id   = astra.create_projector(proj_model+'_fanflat', self.proj_geom, self.vol_geom)
        else: 
            raise Exception("Projection geometry can only be parallel or fanflat.")
        
        self.sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)
        self.recon_id = astra.data2d.create('-vol', self.vol_geom, 0)
    
    def deallocate(self):
        astra.data2d.delete(self.proj_id)

    def __del__(self):
        self.deallocate()
