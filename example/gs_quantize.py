from pathlib import Path
import numpy as np
from example.gs_read_write import _tqdm

def dir_interval(x, limits):
    """
    Convertion of a value in the interval [limits[0], limits[1]] to
    the interval [0,1], (clipping values outside the interval)
    """
    return np.clip((x-limits[0])/(limits[1]-limits[0]), 0, 1)

def inv_interval(x, limits):
    """
    Direct convertion of a value in the interval [0,1] back to the
    interval [limits[0], limits[1]]
    """
    return x*(limits[1]-limits[0])+limits[0]

def quantize(x, bits):
    """
    Quantize a value in the interval [0,1) using 'bits' bits
    """
    q = np.round(x * (2**bits))
    q = np.clip(q, 0, (2**bits) - 1)
    return q

def dequantize(q, bits):
    """
    Dequantize a value represented with 'bits' bits back to the
    inteval [0,1)
    """
    x = q / (2**bits)
    return x

def rgb2yuv(sh_rgb):
    """
    Convert spherical harmonics from RGB to YUV
    """
    sh_yuv = np.zeros(sh_rgb.shape)
    for i in range(sh_rgb.shape[1]):
        sh_yuv[:,i,0] = +0.29900*sh_rgb[:,i,0] + 0.58700*sh_rgb[:,i,1] + 0.11400*sh_rgb[:,i,2]
        sh_yuv[:,i,1] = -0.14713*sh_rgb[:,i,0] - 0.28886*sh_rgb[:,i,1] + 0.43600*sh_rgb[:,i,2]
        sh_yuv[:,i,2] = +0.61500*sh_rgb[:,i,0] - 0.51498*sh_rgb[:,i,1] - 0.10001*sh_rgb[:,i,2]
    return sh_yuv

def yuv2rgb(sh_yuv):
    """
    Convert spherical harmonics from YUV to RGB
    """
    sh_rgb = np.zeros(sh_yuv.shape)
    for i in range(sh_yuv.shape[1]):
        sh_rgb[:,i,0] = sh_yuv[:,i,0] + 1.13983*sh_yuv[:,i,2]
        sh_rgb[:,i,1] = sh_yuv[:,i,0] - 0.39465*sh_yuv[:,i,1] - 0.58060*sh_yuv[:,i,2]
        sh_rgb[:,i,2] = sh_yuv[:,i,0] + 2.03211*sh_yuv[:,i,1]
    return sh_rgb

def dir_pos(pos, start_pos, size_pos):
    """
    Direct transform for the positions
    """
    q_pos = np.zeros(pos.shape)
    for i in range(3):
        q_pos[:,i] = pos[:,i] - start_pos[i]
    q_pos = q_pos/size_pos
    return q_pos

def inv_pos(q_pos, start_pos, size_pos):
    """
    Inverse transform for the positions
    """
    pos = q_pos * size_pos
    for i in range(3):
        pos[:,i] = pos[:,i] + start_pos[i]
    return pos

def quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm=_tqdm):
    """
    Quantize and writes a PLY file to be used by the encoder
    
    Input
    :param bits: Number of bits to use for quantization:
        [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
    
    :param limits: a list containing the limits of the 3DG parameters
        limits = [[start_pos,size_pos], limits_sh, limits_opacity, limits_scale, limits_rot],
        where:
            start_pos: [x,y,z] starting values for the positions
            limits_: [min, max] minimum and maximum value expected for the other attributes
    
    :param pos: 3DG positions
    :type  pos: Nx3 numpy array
    
    :param sh: 3DG spherical harmonics
    :type  sh: NxMx3 numpy array, where M is the number of SH coeffs
    
    :param opacity: 3DG opacity
    :type  opacity: Nx1 numpy array
    
    :param scale: 3DG scale
    :type  scale: Nx3 numpy array
    
    :param rot: 3DG rotation quaternion
    :type  rot: Nx4 numpy array
    
    :param tqdm: function to show a smart progress meter
    
    Output
    :return: quantized positions
    :return: quantized spherical harmonics
    :return: quantized opacity
    :return: quantized scalling
    :return: quantized rotation quaternion
    :return: start positions of the 3DG
    :return: step size for the position
    """
    bits_pos = bits[0]
    bits_sh = bits[1]
    bits_opacity = bits[2]
    bits_scale = bits[3]
    bits_rot = bits[4]
    
    start_pos = limits[0][0]
    size_pos = limits[0][1]
    limits_sh = limits[1]
    limits_opacity = limits[2]
    limits_scale = limits[3]
    limits_rot = limits[4]

    q_pos = dir_pos(pos, start_pos, size_pos)
    q_sh = rgb2yuv(sh)
    q_opacity = opacity
    q_scale = scale
    q_rot = np.zeros(rot.shape)
    for i in tqdm(range(rot.shape[0])):
        norm = np.inner(rot[i,:], rot[i,:])
        if norm > 1e-6:
            if rot[i,0]<0:
                q_rot[i,:] = -rot[i,:]/np.sqrt(norm)
            else:
                q_rot[i,:] = +rot[i,:]/np.sqrt(norm)
        else:
            q_rot[i,:] = [1,0,0,0]
    
    q_pos     = quantize(q_pos, bits_pos)
    q_sh      = quantize(dir_interval(q_sh,           limits_sh), bits_sh)
    q_opacity = quantize(dir_interval(q_opacity, limits_opacity), bits_opacity)
    q_scale   = quantize(dir_interval(q_scale,     limits_scale), bits_scale)
    q_rot     = quantize(dir_interval(q_rot,         limits_rot), bits_rot)

    return q_pos, q_sh, q_opacity, q_scale, q_rot

def dequantize_3dg(bits, limits, q_pos, q_sh, q_opacity, q_scale, q_rot, tqdm=_tqdm):
    """
    Dequantize and writes a PLY file
    
    Input
    :param bits: Number of bits to use for quantization:
        [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
    
    :param limits: a list containing the limits of the 3DG parameters
        limits = [[start_pos,size_pos], limits_sh, limits_opacity, limits_scale, limits_rot],
        where:
            start_pos: [x,y,z] starting values for the positions
            limits_: [min, max] minimum and maximum value expected for the other attributes
    
    :param q_pos: quantized 3DG positions
    :type  q_pos: Nx3 numpy array
    
    :param q_sh: quantized 3DG spherical harmonics
    :type  q_sh: NxMx3 numpy array, where M is the number of SH coeffs
    
    :param q_opacity: quantized 3DG opacity
    :type  q_opacity: Nx1 numpy array
    
    :param q_scale: quantized 3DG scale
    :type  q_scale: Nx3 numpy array
    
    :param q_rot: quantized 3DG rotation quaternion
    :type  q_rot: Nx4 numpy array
    
    :param file: PLY file name
    :type file: Path object
    
    :param tqdm: function to show a smart progress meter
    
    Output
    :return: reconstructed positions
    :return: reconstructed spherical harmonics
    :return: reconstructed opacity
    :return: reconstructed scalling
    :return: reconstructed rotation quaternion
    """
    bits_pos = bits[0]
    bits_sh = bits[1]
    bits_opacity = bits[2]
    bits_scale = bits[3]
    bits_rot = bits[4]
    
    start_pos = limits[0][0]
    size_pos = limits[0][1]
    limits_sh = limits[1]
    limits_opacity = limits[2]
    limits_scale = limits[3]
    limits_rot = limits[4]

    r_pos     = dequantize(q_pos, bits_pos)
    r_sh      = inv_interval(dequantize(q_sh, bits_sh),               limits_sh)
    r_opacity = inv_interval(dequantize(q_opacity, bits_opacity),limits_opacity)
    r_scale   = inv_interval(dequantize(q_scale, bits_scale),      limits_scale)
    r_rot     = inv_interval(dequantize(q_rot, bits_rot),            limits_rot)
    
    for i in tqdm(range(r_rot.shape[0])):
        norm = np.inner(r_rot[i,:], r_rot[i,:])
        if norm > 1e-6:
            if r_rot[i,0]<0:
                r_rot[i,:] = -r_rot[i,:]/np.sqrt(norm)
            else:
                r_rot[i,:] = +r_rot[i,:]/np.sqrt(norm)
        else:
            r_rot[i,:] = [1,0,0,0]
    
    r_pos = inv_pos(r_pos, start_pos, size_pos)
    r_sh = yuv2rgb(r_sh)
    
    return r_pos, r_sh, r_opacity, r_scale, r_rot
