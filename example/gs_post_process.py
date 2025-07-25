import getopt
import sys
from pathlib import Path
from tqdm import tqdm
from gs_quantize import dequantize_3dg
from gs_read_write import readPreprossConfig, read3DG_ply, write3DG_ply

# Modify these locations to reflect your machine
file_quantized = Path("./data/point_cloud_quant.ply")   # output: PLY file of the quantized frame
file_config = Path("./data/point_cloud.json")      # input: json file containing the informarion necessary to inverse the quantization
file_dequantized = Path("./data/point_cloud_dequant.ply") # output: PLY file of the dequantized decoded frame

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:c:o:", ["input_ply=", "config_file=", "--output_ply="])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print("Usage: gs_post_process -i <input_ply_path> -c <config_file_path> -o <output_ply_path>")
        sys.exit()
    elif opt in ("-i", "--input_ply"):
        file_quantized = Path(arg)
    elif opt in ("-c", "--config_file"):
        file_config = Path(arg)
    elif opt in ("-o", "--output_ply"):
        file_dequantized = Path(arg)

# Dequantization
print("read3DG quantized ply")
q_pos, q_sh, q_opacity, q_scale, q_rot = read3DG_ply(file_quantized, tqdm=tqdm)
bits, limits = readPreprossConfig(file_config)

print("dequantize")
r_pos, r_sh, r_opacity, r_scale, r_rot = dequantize_3dg(bits, limits, q_pos, q_sh, q_opacity, q_scale, q_rot, tqdm=tqdm)

print("write3DG dequantized ply")
write3DG_ply(r_pos, r_sh, r_opacity, r_scale, r_rot, True, file_dequantized, tqdm=tqdm)
