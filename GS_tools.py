import os
from pathlib import Path
import filecmp
from tqdm import tqdm
from example.gs_quantize import quantize_3dg, dequantize_3dg
from example.gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply
import subprocess
import numpy as np
import re
import openpyxl
import shutil
from my_tools import File
import multiprocessing
from readme import seq_information,NF
import readme

# 待测试的编解码器,str(tmc3_selected)+'_tmc3.exe'为对应文件
tmc3_selected = {0:"tmc13/anchor",
                 1:"tmc13/MPEG152_rc2_Update_QP",
}


# 输入文件路径
#输出文件名称，对比符号请用__vs__，会自动填写excel内部名称
output_excel="anchor__vs__anchor+aspect1+aspect2.xlsm"
template_excel = f"ctc/{NF}.xlsm"  # 带宏的 Excel 模板


thread_num_limit=[30,3]                     #进程数，建议30个左右，太多容易拥挤，出错

computeMetrics=0
save_iamge=1
save_pointCloud=0
computeSsim=1
computeIvssim=1

cpu=1

# 测试分支，注意：此处请单选
branch_selected = [
    # "octree-predlift",
    # "octree-predlift-inter",
     "octree-raht",
    # "octree-raht-inter",
    # "predgeom-predlift",
    # "predgeom-predlift-inter",
    # "predgeom-raht",
    # "predgeom-raht-inter",
]

# 测试条件，注意：此处请单选
condition_selected = {
     "C1": "lossless-geom-lossy-attrs",
    #"C2": "lossy-geom-lossy-attrs",
    # "CW": "lossless-geom-lossless-attrs",
    # "CY": "lossless-geom-nearlossless-attrs",
}

# 点云类别

class_selected =(
            # "bartender_semitracked",
            #  "cinema_semitracked"   ,
            #  "breakfast_semitracked",
            #  "breakfast_untracked"  ,
            #  "breakdance_untracked" ,
            #  "bartender_tracked"    , #option
            # "cinema_tracked"       , #option
            #  "breakfast_tracked"    ,  #option
             "manwithfruit_tracked" ,
            #  "lego_ferrari"         , #pic big
            #  "lego_bugatti"         ,
            #  "cricket_player"       ,
            #   "plant"                ,
            #  "solo_tango_female"    ,
            #  "solo_tango_male"      ,
            #   "tango_duo"            ,
            #   "tennis_player"        ,
            # "library"              ,#so large
            #   "flowerdance"          ,
            #   "gymnast"              ,
                )

PCC_sequence='/data/Sequence/MPEG_JEE6.1'

file_lock=multiprocessing.Lock()

def pre_process(output,pointCloud):

    file_raw = Path(pointCloud)  # input: the raw model frame in INRIA format
    file_config = Path(output + "/" + "quantized.json")  # output: json file containing the informarion necessary to inverse the quantization
    file_quantized = Path(output + "/" + "quantized.ply")  # output: PLY file with quantized 3DG attributes

    # Modify these to the desired quantization parameters
    bits_pos = 18
    bits_sh = 12
    bits_opacity = 12
    bits_scale = 12
    bits_rot = 12

    limits_pos = [[0, 0, 0], 256]
    limits_sh = [-4, 4]
    limits_opacity = [-7, 18]

    limits_scale = [-26, 4]
    #limits_scale=np.exp(limits_scale)
    limits_rot = [-1, 1]

    bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
    limits = [limits_pos, limits_sh, limits_opacity, limits_scale, limits_rot]

    # Quantization

    pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm)

    for k in range(3):
        limits[0][0][k] = pos[:, k].min()
    writePreprossConfig(file_config, bits, limits)


    q_pos, q_sh, q_opacity, q_scale, q_rot = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm)
    #q_sh[:,1:,:]=2048

    write3DG_ply(q_pos, q_sh, q_opacity, q_scale, q_rot, False, file_quantized, tqdm)

def post_process(output):
    file_decoded = Path(output + "/decoder.ply")  # input: the PLY file of the decoded frame
    file_config = Path(output + "/quantized.json")  # input: json file containing the informarion necessary to inverse the quantization
    file_dequantized = Path(output + "/dequantized.ply")  # output: PLY file of the dequantized decoded frame

    # Dequantization

    q_pos, q_sh, q_opacity, q_scale, q_rot = read3DG_ply(file_decoded, tqdm)
    bits, limits = readPreprossConfig(file_config)


    r_pos, r_sh, r_opacity, r_scale, r_rot = dequantize_3dg(bits, limits, q_pos, q_sh, q_opacity, q_scale, q_rot,
                                                            tqdm)


    write3DG_ply(r_pos, r_sh, r_opacity, r_scale, r_rot, True, file_dequantized, tqdm)

def encoder(output,pointCloud,exe,tmc,isEncoder,branch_selecte):
    def parse_time_output(time_output):
        """解析 /usr/bin/time -v 的输出，提取 MaxRSS"""
        maxrss = None
        for line in time_output.split('\n'):
            if 'Maximum resident set size (kbytes):' in line:
                try:
                    maxrss = int(line.split(':')[1].strip())
                    break
                except (ValueError, IndexError):
                    continue
        return maxrss

    if not os.path.exists(exe):
        print("无启动文件")
    frame=output.split("/")[-1]

    rate_point=output.split("/")[-2]
    os.makedirs(str(Path(output).parent)+"/txt", exist_ok=True)
    main = exe
    condition_selecte = condition_selected[list(condition_selected.keys())[0]]
    cfg_path = tmc3_selected[tmc] + "/cfg_CTC/JEE6.6/" + branch_selecte[0] + "/" + condition_selecte + "/" + rate_point
    para_encfg = "-c " + cfg_path + "/encoder.cfg"
    para_decfg = "-c " + cfg_path + "/decoder.cfg"

    para2 = "--uncompressedDataPath=" + output + "/" + "quantized.ply"
    para3 = "--compressedStreamPath=" + output + "/" + "compress.bin"
    para4 = "--reconstructedDataPath=" + output + "/" + "encoder.ply"


    para = "%s %s %s %s %s" % (main, para_encfg, para2, para3, para4)  # 避免warning输出
    para=f'/usr/bin/time -v {para}'

    if isEncoder:
        _file =str(Path(output).parent)+"/txt/"+frame+"__Bitbream__encoder.txt"
        with open(_file, "w") as f:

            #para = "%s %s %s %s %s" % (main, para_encfg, para2, para3, para4)
            #usage=resource.getrusage(resource.RUSAGE_SELF)

            process = subprocess.Popen(
                para,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 等待进程结束
            out, err = process.communicate()


            # 获取子进程的峰值内存
            peak_mem_kb = parse_time_output(err)
            print(out, file=f)
            print(f"峰值内存: {peak_mem_kb} KB",file=f)

            if not os.path.exists(output + "/" + "encoder.ply"):
                print("编码端重建失败")
                print("运行配置为: " + para)

    else:
        _file = str(Path(output).parent)+"/txt/"+frame+ "__Bitbream__decoder.txt"
        with open(_file, "w") as f:
            para4 = "--reconstructedDataPath=" + output + "/" + "decoder.ply"
            para = "%s %s %s %s" % (main, para_decfg, para3, para4)
            para = f'/usr/bin/time -v {para}'

            process = subprocess.Popen(
                para,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 等待进程结束
            out, err = process.communicate()

            # 获取子进程的峰值内存

            peak_mem_kb = parse_time_output(err)
            print(out, file=f)
            print(f"峰值内存: {peak_mem_kb} KB", file=f)


            if not os.path.exists(output + "/" + "decoder.ply"):
                print("解码端重建失败")
                print("运行配置为: " + para)
                #print(r.stdout)
        if not filecmp.cmp(output + "/" + "encoder.ply", output + "/" + "decoder.ply", shallow=False):
            print("编解码不匹配")

def cam_to_ply(ply,camDIR,exe,output):
    para1="--input="+ply
    para2=para3=""

    if os.path.exists(camDIR+"/cameras.txt"):
        para2="--camera="+camDIR+"/cameras.txt"
    elif os.path.exists(camDIR+"/cameras.bin"):
        para2 = "--camera=" + camDIR + "/cameras.bin"
    else:
        print("没有camera文件")

    if os.path.exists(camDIR+"/images.txt"):
        para3="--image="+camDIR+"/images.txt"
    elif os.path.exists(camDIR+"/images.bin"):
        para3 = "--image=" + camDIR + "/images.bin"
    else:
        print("没有image文件")

    para4="--output="+output
    para5="--verbose=1"
    para = "%s %s %s %s %s %s" % (exe, para1, para2, para3, para4, para5)
    process = subprocess.Popen(
        para,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # 等待进程结束
    out, err = process.communicate()


def metrics(exe,src,dec,frame_start,frame_num,width,hight):
    if computeMetrics==0:
        return 0

    print("render: "+dec)
    para1="-a "+src
    para2="-b "+dec
    para3="--width="+str(width)+" --height="+str(hight)

    para4="-i "+str(frame_start)+" -f "+str(frame_num)+" --cpu="+str(cpu) #cpu
    #para4 = "-i " + str(frame_start) + " -f " + str(frame_num)

    para5 = ("--useCameraPosition=1"
                 f" -s {save_iamge}"
                 f" --computeSsim={computeSsim}"
                 f" --computeIvssim={computeIvssim}"
                 )
        

    para = "OMP_NUM_THREADS=120 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia %s %s %s %s %s %s" % (exe, para1, para2, para3, para4, para5)
    #print(para)
    process = subprocess.Popen(
            para,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # 等待进程结束
    #print(para)
    out, err = process.communicate()
    metric_path = str(Path(dec).parent.parent)+"/metrics/" +f"__metrics.txt"
    os.makedirs(str(Path(dec).parent.parent)+"/metrics", exist_ok=True)
    with file_lock:
        with open(metric_path, "a") as f:
            print(out,file=f)


    print("success render: "+metric_path)

class Gaussian:
    def __init__(self):
        for tmc in tmc3_selected:
            subprocess.run(["bash", "./tmc13/build.sh", tmc3_selected[tmc]])
            
        self.frames = range(0, 1) if NF=="1F" else range(0, 3) # 起始帧，终止帧,指stable的起始帧与结束帧

# ======================================
# 下面变量代码短期不需要修改

        self.PCC_sequence=PCC_sequence+"/"+NF
        self.cameraPosition="./example/cameraPosition"
        self.mpeg_gsc_metrics = "./example/mpeg-gsc-metrics"
        self.rate_points=["r01","r02","r03","r04","r05"]
  

        # ======================================
#下面变量实时更新
        self.condition_selecte = None
        self.class_selecte = class_selected[0]
        self.metrics_path = dict()
        # 加载带宏的模板文件
        self.out_excel=output_excel
        self.wb =openpyxl.load_workbook(template_excel, keep_vba=True,read_only=False)
        self.branch=branch_selected
        def set_name():
            if self.out_excel.find("__vs__"):
                    self.anchor_name=self.out_excel.split("__vs__")[0]
                    self.test_name=self.out_excel.split("__vs__")[1].split(".")[0]

            self.wb["Summary"].cell(row=3, column=3).value = self.anchor_name
            self.wb["Summary"].cell(row=4, column=3).value = self.test_name

        self.set_name=set_name

    def run(self):
        #清空分支
            if os.path.exists(self.branch[0]):
                shutil.rmtree(self.branch[0])

        # ======================================
        # 编解码

            encoders=[True,False]
            for isEncoder in encoders:
                thread_pool = multiprocessing.Pool(thread_num_limit[0])
                for class_selecte in class_selected:
                        for rate_point in self.rate_points:

                            if not class_selecte in seq_information:
                                continue

                            frames=self.frames
                            frames=[x+seq_information[class_selecte][3] for x in frames]
                            for frame in frames :
                                condition_selecte = condition_selected[list(condition_selected.keys())[0]]

                                if NF=="NF":
                                    pointCloud = self.PCC_sequence + "/" + class_selecte + "/plys/"+ f"/frame_{frame:04d}" + ".ply"
                                    cameras_path = self.PCC_sequence + "/" + class_selecte + "/colmap_data/" + f"frame_{frame:04d}/sparse"
                                    output = self.branch[0] + "/" + condition_selecte + "/" + class_selecte + "/" + rate_point + "/" + f"frame{frame:03d}"
                                elif NF=="1F":
                                    pointCloud = File.get_all_file_from_baseCatalog(".ply",self.PCC_sequence+"/"+class_selecte)
                                    cameras_path = File.get_all_file_from_baseCatalog("cameras.bin",self.PCC_sequence+"/"+class_selecte)
                                    cameras_path=os.path.dirname(cameras_path)

                                #self.sub_run(pointCloud, output, frame, self.tmc13, self.tmc, cameras_path, self.cameraPosition,isEncoder, self.branch)
                                for tmc in tmc3_selected:
                                    tmc13=tmc3_selected[tmc]+"/build/tmc3/tmc3"
                                    out=self.branch[0] + "/"+str(tmc)+"/" + condition_selecte + "/" + class_selecte + "/" + rate_point+f"/frame{frame:03d}"
                                    #self.sub_run(pointCloud, out, frame, tmc13, tmc, cameras_path, self.cameraPosition, isEncoder,self.branch)
                                    thread_pool.apply_async(self.sub_run,args=(pointCloud, out, frame, tmc13, tmc, cameras_path, self.cameraPosition,isEncoder, self.branch))

                thread_pool.close()  # 关闭进程池入口，不再接受新进程插入
                thread_pool.join()  # 主进程阻塞，等待进程池中的所有子进程结束，再继续运行主进程

            thread_pool = multiprocessing.Pool(thread_num_limit[1])
            for class_selecte in class_selected:
                    for rate_point in self.rate_points:
                        frames = self.frames
                        frames = [x + seq_information[class_selecte][3] for x in frames]

                        condition_selecte = condition_selected[list(condition_selected.keys())[0]]

                        for frame_id in frames:
                                for tmc in tmc3_selected:
                                    DIR=f"{self.branch[0]}/{tmc}/{condition_selecte}/{class_selecte}/{rate_point}"
                                    #self.render(frame_id, DIR, self.mpeg_gsc_metrics, class_selecte,view_id)
                                    thread_pool.apply_async(self.render, args=(frame_id, DIR, self.mpeg_gsc_metrics, class_selecte))

            thread_pool.close()  # 关闭进程池入口，不再接受新进程插入
            thread_pool.join()  # 主进程阻塞，等待进程池中的所有子进程结束，再继续运行主进程

            if not save_pointCloud:
                all=File.get_all_file_from_baseCatalog(".ply","./")
                for a in all:
                    os.remove(a)

            self.write_to_excel()      #写入excel

    @staticmethod
    def sub_run(pointCloud,output,frame,tmc13,tmc,cameras_path,cameraPosition,isEncoder,branch_selecte):

        if isEncoder:
            print("正在编码:"+output)
        else:
            print("正在解码:"+output)

        os.makedirs(output, exist_ok=True)

        if isEncoder:
            pre_process(output,pointCloud)  # 预处理
            encoder(output,pointCloud,tmc13,tmc,isEncoder,branch_selecte)  # 编码
        else:
            encoder(output, pointCloud, tmc13, tmc, isEncoder,branch_selecte)  # 编码
            post_process(output)  # 后处理

            src_DIR = os.path.dirname(output) + "/src"
            dec_DIR = os.path.dirname(output) + "/dec"
            os.makedirs(src_DIR, exist_ok=True)
            os.makedirs(dec_DIR, exist_ok=True)

            cam_to_ply(pointCloud,cameras_path,cameraPosition,src_DIR+f"/frame{frame:03d}.ply")
            cam_to_ply(output + "/dequantized.ply", cameras_path, cameraPosition, dec_DIR + f"/frame{frame:03d}.ply")

            shutil.rmtree(output)

            print("结束:" + output)


    @staticmethod
    def render(frame_id,DIR,exe,class_selecte):

        src=DIR+"/src/"+"frame"+"%03d" + ".ply"
        dec=DIR+"/dec/"+"frame"+"%03d" + ".ply"
        metrics(exe,src,dec,frame_id,1,seq_information[class_selecte][0],seq_information[class_selecte][1])



    def write_to_excel(self):
        # ======================================
        # 写入PSNR
        metrics_path=File.get_all_file_from_baseCatalog("__metrics.txt",self.branch[0])
        metrics_data=[]
        for path in metrics_path:
            metrics_data.append(self.extract_metrics(path))

        self.sub_write_to_excel(metrics_data)


        # ======================================
        # 写入编码比特
        # ======================================
        bitstream_files=File.get_all_file_from_baseCatalog("__Bitbream__encoder.txt",self.branch[0])
        for bitstream_file in bitstream_files:

            parsed_data = self.parse_bitstream(bitstream_file)
            aggregated = self.aggregate_attributes(parsed_data)
            self.sub_bitstream_write_to_excel(aggregated,bitstream_file)

        self.set_name()


        self.wb.save(self.branch[0] + "/" + self.out_excel.split("/")[-1])
        self.wb.save("./1F-geo/"+self.out_excel)
        print(f"数据已成功写入 1F-geo/{self.out_excel}")

# ======================================
# 步骤 1：从 metrics.txt 提取数据
# ======================================
    def extract_metrics(self,file_path):
        data=dict()
        PSNR = dict()
        with open(file_path, "r") as f:
            contents = f.readlines()

        rgb=[]
        yuv=[]
        ssim_yuv=[]

        for content in contents:
            if content.find("OM-")>=0:         #跳过OM-PSNE,OM-IVSSIM、、
                continue
            if content.find("Psnr RGB (avg)")>=0:
                rgb.append(float(content.split()[4]))
            elif content.find("Psnr YUV (avg)")>=0:
                yuv.append(float(content.split()[4]))
            elif content.find("SSIM (avg)")>=0:
                ssim_yuv.append(float(content.split()[3]))

        frame_num=len(rgb)
        PSNR["PSNR-RGB"]=sum(rgb)/frame_num
        PSNR["PSNR-YCbCr"]=sum(yuv)/frame_num
        PSNR["SSIM-YCbCr"]=sum(ssim_yuv)/frame_num
        PSNR["MIN_PSNR-RGB"]=min(rgb)
        PSNR["MIN_PSNR-YUV"] = min(yuv)
        PSNR["MIN_SSIM-YUV"] = min(ssim_yuv)

        PSNR["MAX_PSNR-RGB"] = max(rgb)
        PSNR["MAX_PSNR-YUV"] = max(yuv)
        PSNR["MAX_SSIM-YUV"] = max(ssim_yuv)

        data[file_path]=PSNR
        return data


# ======================================
# 步骤 2：写入 Excel .xlsm 文件
# ======================================
    def sub_write_to_excel(self,metrics_data):

        # 按图像编号顺序写入数据
        for path in metrics_data:
            key=list(path.keys())[0]
            labels=key.split("/")
            increase_row=int(labels[-3][1:3])-1
            tmc=int(labels[-6])
            sheet_name = 'Proposal' if tmc else 'Anchor'
            _class = labels[-4]

            row = seq_information[_class][5] +increase_row# 假设数据按顺序排列
            data = path[key]
            ws=self.wb[sheet_name]
            # 写入指标数据（保留原始精度）
            for key in data.keys():
                    ws[f'{readme.PSNR_columns[key]}{row}'].value = data[key]



# ======================================
# 步骤 3：提取属性的编码bit
# ======================================
    def parse_bitstream(self,file_path):
        """使用正则表达式解析比特流文件"""
        attributes = dict()
        attributes["e-gtime"]=0
        attributes["e-atime"]=0
        attributes["d-gtime"]=0
        attributes["d-atime"]=0
        attributes["CustomA"]=0
        attributes["CustomB"] = 0
        pattern = re.compile(r'^(\w+).*bitstream size (\d+) B \((\d+\.\d+) bpp\)')


        with open(file_path, 'r') as f:
            contends=f.readlines()

        for i in range(len(contends)-2):

                line=contends[i]
                next1=contends[i+2]
                if line.find("Custom")>=0:
                    if next1.find("f_dc_")>=0:
                        label=line.split()
                        attributes[label[0]] += float(label[1])



        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    attr = match.group(1)  # 属性名
                    size = int(match.group(2))  # 字节数

                    if attr not in attributes:
                        attributes[attr] = 0
                    attributes[attr] +=  size

        total=0
        for key in attributes.keys():
            total+=attributes[key]

        pattern = re.compile(r'^Total bitstream size (\d+) B')
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    size = int(match.group(1))  # 字节数
                    attributes["Total bitstream size"] = size

        attributes["metadata"] = attributes["Total bitstream size"] - total

        pattern = re.compile(r'^Processing time \(user\): (\d+(?:\.\d+)?) s')
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    encoder_time = float(match.group(1))  # 字节数
                    attributes["encoder Processing time (user):"] = encoder_time

        with open(file_path, 'r') as f:
            for l in f:
                line=l.split()
                if l.find("峰值内存")>=0:
                    attributes["e-MaxRSS"] = int(line[1])
                if len(line)<5:
                    continue
                if line[1]=="processing" and line[2]=="time":
                    if line[0]=="positions":
                        attributes["e-gtime"] += float(line[4])
                    else:
                        attributes["e-atime"]+=float(line[4])




        decode_path=file_path.split("__Bitbream__encoder.txt")[0]+"__Bitbream__decoder.txt"

        pattern = re.compile(r'^Processing time \(user\): (\d+(?:\.\d+)?) s')
        with open(decode_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    decoder_time = float(match.group(1))  # 字节数
                    attributes["decoder Processing time (user):"] = decoder_time

        with open(decode_path, 'r') as f:
            for l in f:
                line=l.split()
                if l.find("峰值内存")>=0:
                    attributes["d-MaxRSS"] = int(line[1])
                if len(line)<5:
                    continue
                if line[1]=="processing" and line[2]=="time":
                    if line[0]=="positions":
                        attributes["d-gtime"] += float(line[4])
                    else:
                        attributes["d-atime"]+=float(line[4])


        return attributes

# ======================================
# 步骤 4：属性聚类
# ======================================
    def aggregate_attributes(self,attrs):
        """聚合属性到指定分类"""
        '''
        # 第一种定义：通过结果推理出的
        sh1=["f_rest_0s","f_rest_1s","f_rest_2s",
             "f_rest_15s","f_rest_16s","f_rest_17s",
             "f_rest_30s","f_rest_31s","f_rest_32s"]
        sh2=["f_rest_3s","f_rest_4s","f_rest_5s","f_rest_6s",
             "f_rest_18s","f_rest_19s","f_rest_20s","f_rest_21s",
             "f_rest_33s","f_rest_34s","f_rest_35s","f_rest_36s"]
        sh3=["f_rest_7s","f_rest_8s","f_rest_9s","f_rest_10s","f_rest_11s","f_rest_12s","f_rest_13s","f_rest_14s",
             "f_rest_22s","f_rest_23s","f_rest_24s","f_rest_25s","f_rest_26s","f_rest_27s","f_rest_28s","f_rest_29s",
             "f_rest_37s","f_rest_38s","f_rest_39s","f_rest_40s","f_rest_41s","f_rest_42s","f_rest_43s","f_rest_44s"]
'''

        # 第二种定义：我认为的
        sh1=["f_rest_0s","f_rest_1s","f_rest_2s",
             "f_rest_15s","f_rest_16s","f_rest_17s",
             "f_rest_30s","f_rest_31s","f_rest_32s"]
        sh2=["f_rest_3s","f_rest_4s","f_rest_5s","f_rest_6s","f_rest_7s",
             "f_rest_18s","f_rest_19s","f_rest_20s","f_rest_21s","f_rest_22s",
             "f_rest_33s","f_rest_34s","f_rest_35s","f_rest_36s","f_rest_37s"]
        sh3=["f_rest_8s","f_rest_9s","f_rest_10s","f_rest_11s","f_rest_12s","f_rest_13s","f_rest_14s",
             "f_rest_23s","f_rest_24s","f_rest_25s","f_rest_26s","f_rest_27s","f_rest_28s","f_rest_29s",
             "f_rest_38s","f_rest_39s","f_rest_40s","f_rest_41s","f_rest_42s","f_rest_43s","f_rest_44s"]


        return {
            "Total":attrs["Total bitstream size"],
            "position": sum(attrs[k] for k in attrs.keys() if k.startswith("positions")),
            "sh0": sum(attrs[k] for k in attrs.keys() if k.startswith("f_dc_")),
            "sh1": sum(attrs[k] for k in attrs.keys() if k in sh1),
            "sh2": sum(attrs[k] for k in attrs.keys() if k in sh2),
            "sh3": sum(attrs[k] for k in attrs.keys() if k in sh3),
            "rotation": sum(attrs[k] for k in attrs.keys() if k.startswith("rot_")),
            "scaling": sum(attrs[k] for k in attrs.keys() if k.startswith("scale_")),
            "opacity": sum(attrs[k] for k in attrs.keys() if k.startswith("opacity")),
            #"metadata": attrs["metadata"],  # 元数据占位符
            "T_Enc":attrs["encoder Processing time (user):"],
            "T_Dec":attrs["decoder Processing time (user):"],
            "G_Enc": attrs["e-gtime"],
            "G_Dec": attrs["d-gtime"],
            "A_Enc": attrs["e-atime"],
            "A_Dec": attrs["d-atime"],
            "maxRSS_Enc":attrs["e-MaxRSS"],
            "maxRSS_Dec": attrs["d-MaxRSS"],
            "CustomA":attrs["CustomA"],
            "CustomB": attrs["CustomB"],
        }

# ======================================
# 步骤 5：编码比特写入
# ======================================
    def sub_bitstream_write_to_excel(self,data,bitstream_file):
        # 列映射关系
        columns = readme.Bitstream_columns
        # 定义数据列的起始位置（根据你的 Excel 模板调整）
        # 示例：假设表头在行1，数据从行2开始
        labels = bitstream_file.split("/")
        increase_row = int(labels[-3][1:3]) - 1
        _class=labels[-4]
        row = seq_information[_class][5] + increase_row  # 假设数据按顺序排列
        tmc=int(labels[-6])
        sheet_name = 'Proposal' if tmc else 'Anchor'

        for key in data.keys():
            ws = self.wb[sheet_name]
            a = ws[f'{columns[key]}{row}'].value
            if a is None:
                ws[f'{columns[key]}{row}'] = 0
                ws[f'{columns[key]}{row}'].value += data[key]
            else:
                ws[f'{columns[key]}{row}'].value += data[key]

    def copy_to_excel(self,other_file, my_file, num0, num1):
            # 加载工作簿
            other = openpyxl.load_workbook(other_file, keep_vba=True, read_only=False)
            my = openpyxl.load_workbook(my_file, keep_vba=True, read_only=False)

            # 确定工作表名称
            sheet_name0 = 'Proposal' if num0 else 'Anchor'
            sheet_name1 = 'Proposal' if num1 else 'Anchor'

            # 获取源工作表和目标工作表
            source_sheet = other[sheet_name0]

            # 只复制单元格值
            for row in source_sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:  # 只复制有值的单元格
                        my[sheet_name1].cell(row=cell.row, column=cell.column, value=cell.value)

            # 确保输出目录存在
            os.makedirs("1F-geo", exist_ok=True)

            # 保存文件
            my.save("1F-geo/" + self.out_excel)


if __name__ == '__main__':
    p = "./1F-geo/"
    g = Gaussian()
    g.run()
    






