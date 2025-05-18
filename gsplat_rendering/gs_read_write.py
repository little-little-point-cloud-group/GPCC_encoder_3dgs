from pathlib import Path
import numpy as np
import struct
import json

def _tqdm(input):
    return input

def writePreprossConfig(file, bits, limits):
    """
    Writes the pre prosessing configuration to a json file
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
    
    data = json.dumps({
        "bits": {"pos": bits_pos, "sh": bits_sh, "opacity": bits_opacity, "scale": bits_scale, "rot": bits_rot},
        "limits": {
            "pos": {"start": (start_pos[0], start_pos[1], start_pos[2]), "size": size_pos},
            "sh": {"min": limits_sh[0], "max": limits_sh[1]},
            "opacity": {"min": limits_opacity[0], "max": limits_opacity[1]},
            "scale": {"min": limits_scale[0], "max": limits_scale[1]},
            "rot": {"min": limits_rot[0], "max": limits_rot[1]},
        }
    }, indent=2)
    with file.open("w") as fd:
        fd.write(data)

def readPreprossConfig(file):
    """
    Reads the pre prosessing configuration from a json file
    """
    if not file.exists():
        raise Exception(f"Unable to open {file}")
    with file.open("r") as fd:
        data = json.load(fd)
    bits = data["bits"]
    bits_pos = bits["pos"]
    bits_sh = bits["sh"]
    bits_opacity = bits["opacity"]
    bits_scale = bits["scale"]
    bits_rot = bits["rot"]
    limits = data["limits"]
    size_pos = limits["pos"]["size"]
    start_pos = limits["pos"]["start"]
    limits_sh = [limits["sh"]["min"], limits["sh"]["max"]]
    limits_opacity = [limits["opacity"]["min"], limits["opacity"]["max"]]
    limits_scale = [limits["scale"]["min"], limits["scale"]["max"]]
    limits_rot = [limits["rot"]["min"], limits["rot"]["max"]]
    
    bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
    limits = [[start_pos,size_pos], limits_sh, limits_opacity, limits_scale, limits_rot]
    return bits, limits

def read3DG_ply(file, tqdm=_tqdm):
    """
    Reads a ply file representing a 3DG using INRIA format
    
    Input
    :param file: file name
    :type  file: Path object
    
    :param tqdm: function to show a smart progress meter
    
    Output
    :return: positions of the 3DG
    :return: spherical harmonics
    :return: opacity
    :return: scalling
    :return: rotation quaternion
    """
    if not file.exists():
        raise Exception(f"Unable to open {file}")
    
    vertex = 0
    faces = 0
    file_format = "format ascii 1.0"
    f_rest_numel = 0
    with file.open("rb") as fd:
        header_index = []
        while True:
            line = fd.readline().decode("utf-8")

            if "element vertex" in line:
                vertex = int(line.split()[2])
            elif "element face" in line:
                faces = int(line.split()[2])
                while not "end_header" in line:
                    line = fd.readline().decode("utf-8")
                break
            elif "format" in line:
                file_format = line.replace('\n','').replace('\r','')
            elif "property" in line:
                data = line.split()
                if data[1]=="float" or data[1]=="float32":
                    format_type = ['f', 4]
                elif data[1]=="double" or data[1]=="float64":
                    format_type = ['d', 8]
                elif data[1]=="char" or data[1]=="int8":
                    format_type = ['b', 1]
                elif data[1]=="uchar" or data[1]=="uint8":
                    format_type = ['B', 1]
                elif data[1]=="short" or data[1]=="int16":
                    format_type = ['h', 2]
                elif data[1]=="ushort" or data[1]=="uint16":
                    format_type = ['H', 2]
                elif data[1]=="int" or data[1]=="int32":
                    format_type = ['i', 4]
                elif data[1]=="uint" or data[1]=="uint32":
                    format_type = ['I', 4]

                if data[2]=="x":
                    header_index.append(["pos", 0, format_type[0], format_type[1], False])
                elif data[2]=="y":
                    header_index.append(["pos", 1, format_type[0], format_type[1], False])
                elif data[2]=="z":
                    header_index.append(["pos", 2, format_type[0], format_type[1], False])
                elif data[2]=="nx":
                    header_index.append(["normal", 0, format_type[0], format_type[1], False])
                elif data[2]=="ny":
                    header_index.append(["normal", 1, format_type[0], format_type[1], False])
                elif data[2]=="nz":
                    header_index.append(["normal", 2, format_type[0], format_type[1], False])
                elif "f_dc" in data[2]:
                    index = int(data[2].split('_')[-1])
                    header_index.append(["sh_dc", index, format_type[0], format_type[1], False])
                elif "f_rest" in data[2]:
                    index = int(data[2].split('_')[-1])
                    header_index.append(["sh_ac", index, format_type[0], format_type[1], False])
                    f_rest_numel = max(f_rest_numel, index+1)
                elif data[2]=="opacity":
                    header_index.append(["opacity", 0, format_type[0], format_type[1], False])
                elif "scale" in data[2]:
                    index = int(data[2].split('_')[-1])
                    header_index.append(["scale", index, format_type[0], format_type[1], False])
                elif "rot" in data[2]:
                    index = int(data[2].split('_')[-1])
                    header_index.append(["rot", index, format_type[0], format_type[1], False])
                else:
                    header_index.append(["", 0, format_type[0], format_type[1], False])
            elif "end_header" in line:
                break

        pos = np.zeros(shape=(vertex,3))
        sh_dc = np.zeros(shape=(vertex,3))
        sh_ac = np.zeros(shape=(vertex,f_rest_numel))
        opacity = np.zeros(shape=(vertex,1))
        scale = np.zeros(shape=(vertex,3))
        rot = np.zeros(shape=(vertex,4))
        
        for n in range(len(header_index)):
            if header_index[n][0]=="pos":
                header_index[n][0] = pos
            elif header_index[n][0]=="sh_dc":
                header_index[n][0] = sh_dc
            elif header_index[n][0]=="sh_ac":
                header_index[n][0] = sh_ac
            elif header_index[n][0]=="scale":
                header_index[n][0] = scale
            elif header_index[n][0]=="rot":
                header_index[n][0] = rot
            elif header_index[n][0]=="opacity":
                header_index[n][0] = opacity
            else:
                header_index[n][4] = True
            if file_format == "format ascii 1.0":
                header_index[n][2] = header_index[n][2]=='f' or header_index[n][2]=='d'
        
        if file_format == "format binary_little_endian 1.0":
            for n in tqdm(range(vertex)):
                for array, k, data_type, byte_count, skip in header_index:
                    value = struct.unpack(data_type, fd.read(byte_count))[0]
                    if skip:
                        continue
                    array[n,k] = value
        elif file_format == "format ascii 1.0":
            for n in tqdm(range(vertex)):
                line = fd.readline().decode("utf-8")
                values = line.split()
                i = 0
                for array, k, isfloat, _, skip in header_index:
                    if skip:
                        i += 1
                        continue
                    if isfloat:
                        value = float(values[i])
                    else:
                        value = int(values[i])
                    array[n,k] = value
        else:
            raise Exception(f"Unsupported {file_format}")
    
    Ncoef = int(f_rest_numel/3)+1
    sh = np.zeros(shape=(vertex, Ncoef, 3))
    for i in range(3):
        sh[:,0,i] = sh_dc[:,i]
        for j in range(1, Ncoef):
            sh[:,j,i] = sh_ac[:,j-1+i*(Ncoef-1)]

    return pos, sh, opacity, scale, rot

def write3DG_ply(pos, sh, opacity, scale, rot, flag_isfloat, file, tqdm=_tqdm):
    """
    Write a 3DG ply file

    Input
    :param pos: 3DG positions
    :type  pos: Nx3 numpy array

    :param sh: 3DG spherical harmonics
    :type  sh: NxMx3 numpy array, where M is the number of coefs

    :param opacity: 3DG opacity
    :type  opacity: Nx1 numpy array

    :param scale: 3DG scale [s0,s1,s2]
    :type  scale: Nx3 numpy array

    :param rot: 3DG rotation (quaternion)
    :type  rot: Nx4 numpy array

    :param flag_isfloat: When True, 3DG parameter will be written as float,
        otherwise as uint16
    :type  flag_isfloat: boolean
    
    :param file: file name
    :type  file: Path object
    
    :param tqdm: function to show a smart progress meter
    """
    vertex = pos.shape[0]
    sh_numel = sh.shape[1]*sh.shape[2]

    if flag_isfloat:
        data_type = "float"
    else:
        data_type = "uint16"

    with file.open("wb") as fd:
        fd.write(str.encode("ply\n"))
        fd.write(str.encode("format binary_little_endian 1.0\n"))
        fd.write(str.encode(f"element vertex {vertex}\n"))
        for p in ["x", "y", "z"]:
            fd.write(str.encode(f"property float {p}\n"))
        if flag_isfloat:
            for p in ["nx", "ny", "nz"]:
                fd.write(str.encode(f"property float {p}\n"))
        for i in range(sh.shape[2]):
            fd.write(str.encode(f"property {data_type} f_dc_{i}\n"))
        for i in range((sh.shape[1]-1)*sh.shape[2]):
            fd.write(str.encode(f"property {data_type} f_rest_{i}\n"))
        fd.write(str.encode(f"property {data_type} opacity\n"))
        for i in range(3):
            fd.write(str.encode(f"property {data_type} scale_{i}\n"))
        for i in range(4):
            fd.write(str.encode(f"property {data_type} rot_{i}\n"))
        fd.write(str.encode("end_header\n"))

        if flag_isfloat:
            for n in tqdm(range(vertex)):
                for i in range(3):
                    fd.write(struct.pack('f', pos[n,i]))
                for i in range(3):
                    fd.write(struct.pack('f', 0))
                for i in range(sh.shape[2]):
                    fd.write(struct.pack('f', sh[n,0,i]))
                for i in range(sh.shape[2]):
                    for j in range(1, sh.shape[1]):
                        fd.write(struct.pack('f', sh[n,j,i]))
                fd.write(struct.pack('f', opacity[n,0]))
                for i in range(3):
                    fd.write(struct.pack('f', scale[n,i]))
                for i in range(4):
                    fd.write(struct.pack('f', rot[n,i]))
        else:
            for n in tqdm(range(vertex)):
                for i in range(3):
                    fd.write(struct.pack('f', pos[n,i]))
                for i in range(sh.shape[2]):
                    fd.write(struct.pack('H', int(sh[n,0,i])))
                for i in range(sh.shape[2]):
                    for j in range(1, sh.shape[1]):
                        fd.write(struct.pack('H', int(sh[n,j,i])))
                fd.write(struct.pack('H', int(opacity[n,0])))
                for i in range(3):
                    fd.write(struct.pack('H', int(scale[n,i])))
                for i in range(4):
                    fd.write(struct.pack('H', int(rot[n,i])))
