import getopt
import sys
from pathlib import Path
from tqdm import tqdm
from gs_quantize import quantize_3dg
from gs_read_write import writePreprossConfig, read3DG_ply, write3DG_ply

# Modify these locations to reflect your machine
file_raw = Path("./data/point_cloud.ply")             # input: the raw PLY file
file_config = Path("./data/point_cloud.json")      # input: json file containing the informarion necessary to inverse the quantization
file_quantized = Path("./data/point_cloud_quant.ply")   # output: PLY file of the quantized frame

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:c:o:", ["input_ply=", "config_file=", "--output_ply="])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print("Usage: gs_pre_process -i <input_ply_path> -c <config_file_path> -o <output_ply_path>")
        sys.exit()
    elif opt in ("-i", "--input_ply"):
        file_raw = Path(arg)
    elif opt in ("-c", "--config_file"):
        file_config = Path(arg)
    elif opt in ("-o", "--output_ply"):
        file_quantized = Path(arg)

# Modify these to the desired quantization parameters
bits_pos = 18
bits_sh = 12
bits_opacity = 12
bits_scale = 12
bits_rot = 12

limits_pos = [[0,0,0], 256]
limits_sh = [-4,4]
limits_opacity = [-7,18]
limits_scale = [-26,4]
limits_rot = [-1,1]

bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
limits = [limits_pos, limits_sh, limits_opacity, limits_scale, limits_rot]

# Quantization
print("read3DG raw ply")
pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm=tqdm)

for k in range(3):
    limits[0][0][k] = pos[:,k].min()
writePreprossConfig(file_config, bits, limits)

print("quantize")
q_pos, q_sh, q_opacity, q_scale, q_rot = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm=tqdm)

print("write3DG quantized ply")
write3DG_ply(q_pos, q_sh, q_opacity, q_scale, q_rot, False, file_quantized, tqdm=tqdm)
