from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from gs_quantize import quantize_3dg, dequantize_3dg
from gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply

# Modify these locations to reflect your machine
base=r'C:\little_boy\vs_workspace\3dgs\mpeg-pcc-tmc13\build\test'
file_decoded = Path(base+"/file.ply")         # input: the PLY file of the decoded frame
file_config = Path(base+"/file.json")      # input: json file containing the informarion necessary to inverse the quantization
file_dequantized = Path(base+"/defile.ply") # output: PLY file of the dequantized decoded frame

# Dequantization
print("-( Read PLY )-------------------------")
q_pos, q_sh, q_opacity, q_scale, q_rot = read3DG_ply(file_decoded, tqdm)
bits, limits = readPreprossConfig(file_config)
    
print("-( Dequantize )-----------------------")
r_pos, r_sh, r_opacity, r_scale, r_rot = dequantize_3dg(bits, limits, q_pos, q_sh, q_opacity, q_scale, q_rot, tqdm)

print("-( Write dequantized PC )-------------")
write3DG_ply(r_pos, r_sh, r_opacity, r_scale, r_rot, True, file_dequantized, tqdm)
