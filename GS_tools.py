import os
from pathlib import Path

from tqdm import tqdm
from example.gs_quantize import quantize_3dg, dequantize_3dg
from example.gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply
import subprocess
import matplotlib.pyplot as plt
import lpips
import numpy as np
import re
import openpyxl
import shutil
from my_tools import File
from gsplat_rendering.gs_render_frames import render
import multiprocessing


# 测试分支，注意：此处请单选
branch_selected = (
    # "octree-predlift",
    # "octree-predlift-inter",
    "octree-raht",
    # "octree-raht-inter",
    # "predgeom-predlift",
    # "predgeom-predlift-inter",
    # "predgeom-raht",
    # "predgeom-raht-inter",
)
# 测试条件，注意：此处请单选
condition_selected = {
     "C1": "lossless-geom-lossy-attrs",
    #"C2": "lossy-geom-lossy-attrs",
    # "CW": "lossless-geom-lossless-attrs",
    # "CY": "lossless-geom-nearlossless-attrs",
}

# 点云类别，目前这个变量没有使用
class_selected = (
    "m71763_breakfast_stable",
    #"m71763_cinema_stable",
)

tracks=(
    "track",
    #"partially-track"
)

# 待测试的编解码器,0代表anchor，1代表test,支持选择一个
tmc3_selected = 1


# 输入文件路径
template_excel = "ctc/anchor.xlsm"  # 带宏的 Excel 模板
thread_num_limit=2


PCC_sequence=r'C:\little_boy\PCC_sequence\3DGS'


def pre_process(output,class_selecte,track,frame):
    # Modify these locations to reflect your machine
    pointCloud=PCC_sequence+"/"+class_selecte+"/"+track+"/"+frame+".ply"
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
    limits_rot = [-1, 1]

    bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
    limits = [limits_pos, limits_sh, limits_opacity, limits_scale, limits_rot]

    # Quantization

    pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm)

    for k in range(3):
        limits[0][0][k] = pos[:, k].min()
    writePreprossConfig(file_config, bits, limits)


    q_pos, q_sh, q_opacity, q_scale, q_rot = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm)


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


def encoder(output,rate_point):
    _file =output+"/"+rate_point+"__Bitbream__encoder.txt"
    with open(_file, "w") as f:
        main = str(tmc3_selected)+'_tmc3.exe'
        condition_selecte = condition_selected[list(condition_selected.keys())[0]]
        cfg_path = os.getcwd() + "/cfg" + str(tmc3_selected) + "/" + branch_selected[0] + "/" + condition_selecte + "/" + rate_point
        para_encfg = "-c " + cfg_path + "/encoder.cfg"
        para_decfg = "-c " + cfg_path + "/decoder.cfg"

        para2 = "--uncompressedDataPath=" + output + "/" + "quantized.ply"
        para3 = "--compressedStreamPath=" + output + "/" + "compress.bin"
        #para4 = "--reconstructedDataPath=" + output + "/" + "encoder.ply"
        para = "%s %s %s %s" % (main, para_encfg, para2, para3)
        r = subprocess.run(para, capture_output=True, text=True)
        print(r.stdout, file=f)
        if not os.path.exists(output + "/" + "encoder.ply"):
            pass
            #print("编码端重建失败")
            #print("运行配置为: " + para)

    _file = output + "/" + rate_point + "__Bitbream__decoder.txt"
    with open(_file, "w") as f:
        para4 = "--reconstructedDataPath=" + output + "/" + "decoder.ply"
        para = "%s %s %s %s" % (main, para_decfg, para3, para4)
        r = subprocess.run(para, capture_output=True, text=True)
        print(r.stdout, file=f)
        if not os.path.exists(output + "/" + "decoder.ply"):
            print("解码端重建失败")
            print("运行配置为: " + para)
            print(r.stdout)


def metrics(output,class_selecte,track,frame):
    main = "ctc/QMIV.exe"
    gt = File.get_all_file_from_baseCatalog(".png", PCC_sequence+"/"+class_selecte+"/renders/"+track+"/"+frame)
    render = File.get_all_file_from_baseCatalog(".png", output + "/renders" )
    metric_path = output + "/__metrics.txt"

    with open(metric_path, "w") as f:
        for i in range(len(gt)):
            # 测试rgb下PSNR
            para1 = "-i0 " + gt[i]
            para2 = "-i1 " + render[i]
            para3 = "-ml PSNR"

            image = plt.imread(gt[i])
            height, width = image.shape[:2]
            para4 = "-ps " + str(width) + "x" + str(height)
            para5 = "-ff PNG"
            para6 = "-cwa " + "1:1:1:0"
            para7 = "-cws " + "1:1:1:0"
            para = "%s %s %s %s %s %s %s %s" % (main, para1, para2, para3, para4, para5, para6, para7)

            r = subprocess.run(para, capture_output=True, text=True)
            print(r.stdout, file=f)

            # 测试yuv下PSNR,SSIM,IVSSIM

            para3 = "-ml PSNR,SSIM,IVSSIM"

            para6 = "-csm YCbCr_BT709"
            para = "%s %s %s %s %s %s %s" % (main, para1, para2, para3, para4, para5, para6)
            r = subprocess.run(para, capture_output=True, text=True)
            print(r.stdout, file=f)

            # 计算LPIPPS
            loss = lpips.LPIPS(net='alex',verbose=False)
            img0 = lpips.im2tensor(lpips.load_image(gt[i]))
            img1 = lpips.im2tensor(lpips.load_image(render[i]))
            d = loss(img0, img1)
            print(">>>> LPIPS:" + str(np.array(d.data.cpu())[0, 0, 0, 0]), file=f)




class Gaussian:
    def __init__(self,PCC_sequence=PCC_sequence):
# ======================================
# 下面变量代码为测试的运行文件
        self.tmc13=str(tmc3_selected)+'_tmc3.exe'

# ======================================
# 下面变量代码短期不需要修改

        print("代码："+self.tmc13)
        self.PCC_sequence=PCC_sequence
        self.output_excel=branch_selected[0]+".xlsm"
        #self.rate_points=["r01","r02","r03","r04",]
        self.rate_points=["r02",]

        self.anchor_columns = {
            "PSNR-RGB": "F",  # PSNR-RGB 列
            "PSNR-YCbCr": "G",  # PSNR-YCbCr 列
            "SSIM-YCbCr": "H",  # SSIM-YCbCr 列
            "IVSSIM": "I",  # IVSSIM 列
            "LPIPS": "J"  # LPIPS 列
        }

        self.test_columns = {
            "PSNR-RGB": "X",  # PSNR-RGB 列
            "PSNR-YCbCr": "Y",  # PSNR-YCbCr 列
            "SSIM-YCbCr": "Z",  # SSIM-YCbCr 列
            "IVSSIM": "AA",  # IVSSIM 列
            "LPIPS": "AB"  # LPIPS 列
        }

        self.PSNR_start_row = 33

        self.anchor_Bitstream_columns={
        "position":"E",
        "sh0":"F",
        "sh1":"G",
        "sh2":"H",
        "sh3":"I",
        "rotation":"J",
        "scaling":"K",
        "opacity":"L",
        "metadata":"M",
        }

        self.test_Bitstream_columns={
        "position":"W",
        "sh0":"X",
        "sh1":"Y",
        "sh2":"Z",
        "sh3":"AA",
        "rotation":"AB",
        "scaling":"AC",
        "opacity":"AD",
        "metadata":"AE",
        }


        # 复制工作表
        self.tracks_name = {
            "track": "track",
            "partially-track": "semitrack",
        }

        self.Bitstream_start_row = 16
# ======================================
#下面变量实时更新
        self.condition_selecte = None
        self.frames=30             #32帧数据
        self.class_selecte = class_selected[0]
        self.metrics_path = dict()
        self.tmc=tmc3_selected
        # 加载带宏的模板文件
        self.wb =openpyxl.load_workbook(template_excel, keep_vba=True,read_only=False)

    def run(self):
        #清空分支
            if os.path.exists(branch_selected[0]):
                shutil.rmtree(branch_selected[0])

        # ======================================
        # 下面为检测anchor的render图像，没有的话就立即渲染
            self.image_anchor()

            thread_pool = multiprocessing.Pool(thread_num_limit)
            for class_selecte in class_selected:
                for track in tracks:
                    for rate_point in self.rate_points:
                        for frame in range(self.frames):
                            #self.sub_run(class_selecte, track, rate_point, frame)
                            thread_pool.apply_async(self.sub_run, args=(class_selecte,track,rate_point,frame))


            thread_pool.close()  # 关闭进程池入口，不再接受新进程插入
            thread_pool.join()  # 主进程阻塞，等待进程池中的所有子进程结束，再继续运行主进程

            self.write_to_excel()      #写入excel

    @staticmethod
    def sub_run(class_selecte,track,rate_point,frame):
        condition_selecte=condition_selected[list(condition_selected.keys())[0]]
        output =  os.path.join(os.getcwd(),branch_selected[0],condition_selecte,class_selecte,track,rate_point,f"frame{frame:03d}")
        print("正在运行:"+os.path.join(branch_selected[0],condition_selecte,class_selecte,track,rate_point,f"frame{frame:03d}"))
        os.makedirs(output, exist_ok=True)
        frame=f"frame{frame:03d}"
        pre_process(output,class_selecte,track,frame)  # 预处理
        encoder(output,rate_point)  # 编码
        post_process(output)  # 后处理

        cameras_path=PCC_sequence+"/"+class_selecte+"/cameras/"+frame
        ply_path=output+"/dequantized.ply"
        output_dir=output+"/renders"
        render(cameras_path,ply_path,output_dir)  # 渲染
        metrics(output,class_selecte,track,frame)  # 计算失真
        print("结束:" + os.path.join(branch_selected[0], condition_selecte, class_selecte, track, rate_point,
                                         f"frame{frame:03d}"))

    def image_anchor(self):

        for class_selecte in class_selected:
            class_path=self.PCC_sequence+"/"+class_selecte+"/renders"
            for track in tracks:
                track_path=class_path+"/"+track
                plys=os.listdir(self.PCC_sequence+"/"+class_selecte+"/"+track)
                cameras=os.listdir(self.PCC_sequence+"/"+class_selecte+"/cameras")
                for i in range(len(plys)):
                    if not os.path.exists(track_path+"/"+cameras[i]):
                        os.makedirs(track_path+"/"+cameras[i], exist_ok=True)
                        print(f"正在渲染:{class_selecte}/{track}/{plys[i]}")
                        print(f"相机路径:cameras/{cameras[i]}")
                        render(self.PCC_sequence+"/"+class_selecte+"/cameras/"+cameras[i],self.PCC_sequence+"/"+class_selecte+"/"+track+"/"+plys[i],track_path+"/"+cameras[i])

    def write_to_excel(self):

        # ======================================
        # 创建sheet
        self.create_sheet()


        # ======================================
        # 写入PSNR
        # ======================================
        metrics_path=File.get_all_file_from_baseCatalog("__metrics.txt",branch_selected[0])
        metrics_data=dict()
        for path in metrics_path:
            metrics_data[path]=self.extract_metrics(path)

        self.sub_write_to_excel(metrics_data)


        # ======================================
        # 写入编码比特
        # ======================================
        bitstream_files=File.get_all_file_from_baseCatalog("__Bitbream__encoder.txt",branch_selected[0])
        for bitstream_file in bitstream_files:

            labels=bitstream_file.split("/")
            increase_row=int(labels[-1][1:3])-1

            parsed_data = self.parse_bitstream(bitstream_file)
            aggregated = self.aggregate_attributes(parsed_data)
            self.sub_bitstream_write_to_excel(aggregated,bitstream_file)


        # 保存为新文件（保留宏）
        self.wb.save(branch_selected[0]+"/"+self.output_excel)
        print(f"数据已成功写入 {self.output_excel}")
        # 保存为新文件（保留宏）
        os.makedirs("result", exist_ok=True)
        self.wb.save("result/"+self.output_excel)


# ======================================
# 步骤 1：从 metrics.txt 提取数据
# ======================================
    def extract_metrics(self,file_path):
        with open(file_path, "r") as f:
            content = f.read()

        # 正则表达式匹配关键指标（优化后的表达式）
        pattern = re.compile(
            r"Average\s+PSNR-RGB\s+([\d.]+).*?"  # PSNR-RGB
            r"Average\s+PSNR-YCbCr\s+([\d.]+).*?"  # PSNR-YCbCr
            r"Average\s+SSIM-YCbCr\s+([\d.]+).*?"  # SSIM-YCbCr
            r"Average\s+IVSSIM\s+([\d.]+).*?"  # IVSSIM
            r">>>> LPIPS:([\d.]+)",  # LPIPS
            re.DOTALL
        )
        matches = pattern.findall(content)

        # 数据整理为字典格式 {图像编号: 指标}
        data = {}
        frame_id=0
        for match in matches:
             # 00000 -> 0, 00001 -> 1
            data[f"v__{frame_id}"] = {
                "PSNR-RGB": float(match[0]),
                "PSNR-YCbCr": float(match[1]),
                "SSIM-YCbCr": float(match[2]),
                "IVSSIM": float(match[3]),
                "LPIPS": float(match[4])
            }
            frame_id=frame_id+1
        return data


# ======================================
# 步骤 2：写入 Excel .xlsm 文件
# ======================================
    def sub_write_to_excel(self,metrics_data):

        columns=self.test_columns if self.tmc else self.anchor_columns

        # 定义数据列的起始位置（根据你的 Excel 模板调整）
        # 示例：假设表头在行1，数据从行2开始
        start_row = self.PSNR_start_row
        # 按图像编号顺序写入数据
        for path in metrics_data.keys():
            labels=path.split("/")
            increase_row=int(labels[-3][1:3])-1

            _class = labels[-5].split("_")[1]
            track=labels[-4]
            sheet_name = f'{_class}_{self.tracks_name[track]}'

            for view in sorted(metrics_data[path].keys()):
                row = start_row + 5*int(view[3:])+increase_row# 假设数据按顺序排列
                data = metrics_data[path][view]
                ws=self.wb[sheet_name]
                # 写入指标数据（保留原始精度）
                for key in columns.keys():
                    a=ws[f'{columns[key]}{row}'].value
                    if a is None:
                        ws[f'{columns[key]}{row}']=0
                        ws[f'{columns[key]}{row}'].value += data[key]
                    else:
                        ws[f'{columns[key]}{row}'].value += data[key]

        for class_selecte in class_selected:
            for track in tracks:
                    _class=class_selecte.split("_")[1]
                    sheet_name = f'{_class}_{self.tracks_name[track]}'
                    ws = self.wb[sheet_name]
                    render_path = self.PCC_sequence + "/" + class_selecte + "/renders/"+track+"/frame000"
                    for r in range(len(self.rate_points)):
                        for v in range(len(render_path)):
                            row = start_row + 5 * v + r  # 假设数据按顺序排列
                            for key in columns.keys():

                                a = ws[f'{columns[key]}{row}'].value
                                if a is None:
                                   pass
                                else:
                                    ws[f'{columns[key]}{row}'].value /=self.frames



# ======================================
# 步骤 3：提取属性的编码bit
# ======================================
    def parse_bitstream(self,file_path):
        """使用正则表达式解析比特流文件"""
        attributes = dict()
        pattern = re.compile(r'^(\w+).*bitstream size (\d+) B \((\d+\.\d+) bpp\)')

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
        return attributes

# ======================================
# 步骤 4：属性聚类
# ======================================
    def aggregate_attributes(self,attrs):
        """聚合属性到指定分类"""
        #第一中定义：我认为的
        '''
        sh1=["f_rest_0s","f_rest_1s","f_rest_2s",
             "f_rest_15s","f_rest_16s","f_rest_17s",
             "f_rest_30s","f_rest_31s","f_rest_32s"]
        sh2=["f_rest_3s","f_rest_4s","f_rest_5s","f_rest_6s","f_rest_7s",
             "f_rest_18s","f_rest_19s","f_rest_20s","f_rest_21s","f_rest_22s",
             "f_rest_33s","f_rest_34s","f_rest_35s","f_rest_36s","f_rest_37s"]
        sh3=["f_rest_8s","f_rest_9s","f_rest_10s","f_rest_11s","f_rest_12s","f_rest_13s","f_rest_14s",
             "f_rest_23s","f_rest_24s","f_rest_25s","f_rest_26s","f_rest_27s","f_rest_28s","f_rest_29s",
             "f_rest_38s","f_rest_39s","f_rest_40s","f_rest_41s","f_rest_42s","f_rest_43s","f_rest_44s"]
        '''
        #第二种定义：通过结果推理出的（大概率这种是正确的，但不清楚）
        sh1=[]
        sh2=[]
        sh3=[]
        for i in range(9):
            sh1.append(f"f_rest_{str(i)}s")
        for i in range(9,24):
            sh2.append(f"f_rest_{str(i)}s")
        for i in range(24,45):
            sh3.append(f"f_rest_{str(i)}s")

        return {
            "position": sum(attrs[k] for k in attrs.keys() if k.startswith("positions")),
            "sh0": sum(attrs[k] for k in attrs.keys() if k.startswith("f_dc_")),
            "sh1": sum(attrs[k] for k in attrs.keys() if k in sh1),
            "sh2": sum(attrs[k] for k in attrs.keys() if k in sh2),
            "sh3": sum(attrs[k] for k in attrs.keys() if k in sh3),
            "rotation": sum(attrs[k] for k in attrs.keys() if k.startswith("rot_")),
            "scaling": sum(attrs[k] for k in attrs.keys() if k.startswith("scale_")),
            "opacity": sum(attrs[k] for k in attrs.keys() if k.startswith("opacity")),
            "metadata": 0  # 元数据占位符
        }

# ======================================
# 步骤 5：编码比特写入
# ======================================
    def sub_bitstream_write_to_excel(self,data,bitstream_file):
        labels=bitstream_file.split("/")

        # 列映射关系
        columns = self.test_Bitstream_columns if tmc3_selected else self.anchor_Bitstream_columns

        # 定义数据列的起始位置（根据你的 Excel 模板调整）
        # 示例：假设表头在行1，数据从行2开始
        start_row = 16
        labels = bitstream_file.split("/")
        increase_row = int(labels[-3][1:3]) - 1

        _class = labels[-5].split("_")[1]
        track = labels[-4]
        sheet_name = f'{_class}_{self.tracks_name[track]}'

        row = start_row + increase_row  # 假设数据按顺序排列
        for key in data.keys():
            value = data[key]
            ws = self.wb[sheet_name]
            a = ws[f'{columns[key]}{row}'].value
            if a is None:
                ws[f'{columns[key]}{row}'] = 0
                ws[f'{columns[key]}{row}'].value += data[key] * 8 / 1000
            else:
                ws[f'{columns[key]}{row}'].value += data[key] * 8 / 1000


    def create_sheet(self,anchor="G-PCC v1(151 config)",test="my"):
        row=4
        row_main=9
        # 获取源工作表
        source_sheet = self.wb['M_NC5']

        self.wb["Summary (auto)"].cell(row=3, column=3).value = anchor
        self.wb["Summary (auto)"].cell(row=4, column=3).value = test

        sheets=["RGB-PSNR","YUV-PSNR","YUV-SSIM","YUV-IVSSIM","LPIPS"]
        for class_selecte in class_selected:
            for track in tracks:
                new_sheet = self.wb.copy_worksheet(source_sheet)
                _class=class_selecte.split("_")[1]
                sheet_name=f'{_class}_{self.tracks_name[track]}'
                new_sheet.title = sheet_name
                new_sheet.cell(row=2,column=2).value =sheet_name
                # 写入指标数据（保留原始精度）
                for sheet in sheets:
                    self.wb[sheet].cell(row=row,column=2).value = sheet_name

                self.wb["Summary (auto)"].cell(row=row_main,column=3).value =sheet_name
                row=row+11
                row_main=row_main+1
        self.wb.remove(self.wb['M_NC5'])


if __name__ == '__main__':
    g = Gaussian()
    g.run()