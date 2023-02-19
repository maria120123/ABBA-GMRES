# %% ***************************************************** Import Packages *****************************************************
import numpy as np
import astra 
import tigre

# %% ********************************************************** TIGRE **********************************************************
class fp_tigre:
    '''forward_projection
    Forward projection models: 'Siddon' (Line) or 'interpolated' (Joseph)
    '''

    def __init__(self,ct_tigre):
        self.geo = ct_tigre.geo
        self.N_det = ct_tigre.N_det
        self.N_ang = ct_tigre.N_ang
        self.angles = ct_tigre.proj_angles
        self.mode = ct_tigre.proj_geom
        self.fp_model = ct_tigre.fp_model

    def apply_A(self,x):
        sinogram = tigre.Ax(x.reshape(self.geo.nVoxel),self.geo,self.angles,projection_type=self.fp_model,mode=self.mode).reshape(self.N_ang,self.N_det)
        return sinogram.reshape(-1)

    def __matmul__(self,x):
        return self.apply_A(x)

class bp_tigre:
    '''back_projection
    Back projection models: 'matched' or 'FDK'
    '''
    def __init__(self,ct_tigre):
        self.geo = ct_tigre.geo
        self.angles = ct_tigre.proj_angles
        self.N_det = ct_tigre.N_det
        self.N_ang = ct_tigre.N_ang
        self.N = ct_tigre.N
        self.mode = ct_tigre.proj_geom
        self.bp_model = ct_tigre.bp_model

    def apply_B(self,b):
        Bb = tigre.Atb(b.reshape(self.N_ang, 1, self.N_det), self.geo, self.angles, backprojection_type=self.bp_model,mode=self.mode).reshape(self.N,self.N)
        return Bb.reshape(-1)

    def __matmul__(self,b):
        return self.apply_B(b)

# %% ********************************************************** ASTRA **********************************************************
class fp_astra:
    '''forward projection
    Forward projection models: 'line', 'strip', or 'linear' (Joseph)
    '''
    def __init__(self,ct_astra):
        self.N = ct_astra.N
        self.N_ang = ct_astra.N_ang
        self.N_det = ct_astra.N_det 
        self.proj_id = ct_astra.proj_id
        self.proj_geom = ct_astra.proj_geom
        self.GPU = ct_astra.GPU
        self.vol_geom = ct_astra.vol_geom

    def apply_A(self,x):
        volume_id = astra.data2d.create('-vol', self.vol_geom, x.reshape(self.N,self.N))
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)
        if self.GPU == True:
            cfg = astra.creators.astra_dict('FP_CUDA')
        else:
            cfg = astra.creators.astra_dict('FP')
        cfg['ProjectorId'] = self.proj_id       # Id to forward projector A
        cfg['ProjectionDataId'] = sinogram_id   # Id to sinogram b (what A should be multiplied with)
        cfg['VolumeDataId'] = volume_id         # Id to size/volume of output
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete(alg_id)
        sinogram = astra.data2d.get(sinogram_id)

        return sinogram.reshape(-1)

    def __matmul__(self,x):
        return self.apply_A(x)

class bp_astra:
    '''back projection'''
    def __init__(self,ct_astra):
        self.N_ang = ct_astra.N_ang
        self.N_det = ct_astra.N_det 
        self.vol_geom = ct_astra.vol_geom
        self.proj_id = ct_astra.proj_id
        self.proj_geom = ct_astra.proj_geom
        self.GPU = ct_astra.GPU
        self.sinogram_id = ct_astra.sinogram_id

    def apply_B(self,b):
        if self.GPU == True:
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, b.reshape(self.N_ang,self.N_det))
            recon_id = astra.data2d.create('-vol', self.vol_geom, 0)
            cfg = astra.creators.astra_dict('BP_CUDA')
            cfg['ProjectorId'] = self.proj_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['ReconstructionDataId'] = recon_id
            bp_id = astra.algorithm.create(cfg)
            astra.algorithm.run(bp_id)
            Bb = astra.data2d.get(recon_id)
        else:
            _, Bb = astra.create_backprojection(b.reshape(self.N_ang,self.N_det), self.proj_id)
        return Bb.reshape(-1)

    def __matmul__(self,b):
        return self.apply_B(b)
