import os
from pathlib import Path
from tqdm import tqdm
from example.gs_quantize import quantize_3dg, dequantize_3dg
from example.gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply
import subprocess
import sys
import matplotlib.pyplot as plt
import lpips
import numpy as np
import re
import openpyxl
import shutil
from my_tools import File
from ctc.data_name import Mandatory_new



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
# 测试条件
condition_selected = {
     "C1": "lossless-geom-lossy-attrs",
    #"C2": "lossy-geom-lossy-attrs",
    # "CW": "lossless-geom-lossless-attrs",
    # "CY": "lossless-geom-nearlossless-attrs",
}

# 点云类别，目前这个变量没有使用
class_selected = (
    "Cinema",
    #"Bartender",
    #"Cinema",
    #"Breakfast",
    #"ManWithFruit",
)

# 待测试的编解码器,0代表anchor，1代表test，目前只能选择一个
tmc3_selected = 0

# 输入文件路径
template_excel = "ctc/excel_template_GSC.xlsm"  # 带宏的 Excel 模板

class Gaussian:
    def __init__(self,base):
        if os.path.exists("data"):
            shutil.rmtree("data")  # 递归删除目标文件夹
        shutil.copytree(base,"data")
# ======================================
# 下面变量代码为测试的运行文件
        self.tmc13=str(tmc3_selected)+'_tmc13.exe'

# ======================================
# 下面变量代码短期不需要修改

        print("代码："+self.tmc13)
        self.base=os.getcwd()+"/data"
        self.pointCloud=base+"/point_cloud/iteration_30000/point_cloud.ply"
        self.output_excel=branch_selected[0]+".xlsm"
        self.rate_points=[
            "r01","r02","r03","r04","r05"
        ]

        self.anchor_columns = {
            "ID": "B",  # 图像编号列（如 v00）
            "PSNR-RGB": "F",  # PSNR-RGB 列
            "PSNR-YCbCr": "G",  # PSNR-YCbCr 列
            "SSIM-YCbCr": "H",  # SSIM-YCbCr 列
            "IVSSIM": "I",  # IVSSIM 列
            "LPIPS": "J"  # LPIPS 列
        }

        self.test_columns = {
            "ID": "T",  # 图像编号列（如 v00）
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

        self.Bitstream_start_row = 16
# ======================================
#下面变量实时更新
        self.condition_selecte = None
        self.rate_point = None
        self.class_selecte = class_selected[0]
        self.metrics_path = dict()
        self.tmc = None

        # 加载带宏的模板文件
        self.wb =openpyxl.load_workbook(template_excel, keep_vba=True,read_only=False)

    def run(self):
        #清空分支
            if os.path.exists(branch_selected[0]):
                shutil.rmtree(branch_selected[0])
            if os.path.exists("image"):
                shutil.rmtree("image")


            self.tmc=tmc3_selected
            for condition_selecte in condition_selected:
                self.condition_selecte=condition_selecte
                for class_selecte in class_selected:
                    self.class_selecte=class_selecte
                    for rate_point in self.rate_points:
                        self.rate_point=rate_point
                        self.output=os.getcwd()+"/"+branch_selected[0]+"/"+self.condition_selecte+"/"+self.class_selecte+"/"+rate_point
                        os.makedirs(self.output,exist_ok=True)

                        self.image_anchor()  # 生成对比图像
                        self.pre_process()         #预处理
                        self.encoder()             #编码
                        self.post_process()        #后处理
                        self.render()              #渲染
                        self.metrics()             #计算失真

            self.write_to_excel()      #写入excel

    def image_anchor(self):
        # 切换到目标目录
        if os.path.exists("image"+"/"+self.class_selecte):
            return 0

        os.makedirs("image", exist_ok=True)
        os.chdir("gaussiansplatting")

        # 执行命令：python convert.py -s data
        cmd = [sys.executable, 'render.py', '-m', self.base]
        subprocess.run(cmd, check=True)
        os.chdir("..")
        shutil.move(self.base + "/train/ours_30000/renders", "image")

        os.rename("image/renders","image"+"/"+self.class_selecte)

    def pre_process(self):
        # Modify these locations to reflect your machine

        file_raw = Path(self.pointCloud)# input: the raw model frame in INRIA format
        file_config = Path(self.output+"/"+"quantized.json")  # output: json file containing the informarion necessary to inverse the quantization
        file_quantized = Path(self.output+"/"+"quantized.ply")  # output: PLY file with quantized 3DG attributes



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
        print("-( Read PLY )-------------------------")
        pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm)

        for k in range(3):
            limits[0][0][k] = pos[:, k].min()
        writePreprossConfig(file_config, bits, limits)

        print("-( Quantize )-------------------------")
        q_pos, q_sh, q_opacity, q_scale, q_rot = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm)

        print("-( Write quantized PC )---------------")
        write3DG_ply(q_pos, q_sh, q_opacity, q_scale, q_rot, False, file_quantized, tqdm)

    def post_process(self):

        file_decoded = Path(self.output+"/decoder.ply")  # input: the PLY file of the decoded frame
        file_config = Path(self.output+"/quantized.json")  # input: json file containing the informarion necessary to inverse the quantization
        file_dequantized = Path(self.output+"/dequantized.ply")  # output: PLY file of the dequantized decoded frame

        # Dequantization
        print("-( Read PLY )-------------------------")
        q_pos, q_sh, q_opacity, q_scale, q_rot = read3DG_ply(file_decoded, tqdm)
        bits, limits = readPreprossConfig(file_config)

        print("-( Dequantize )-----------------------")
        r_pos, r_sh, r_opacity, r_scale, r_rot = dequantize_3dg(bits, limits, q_pos, q_sh, q_opacity, q_scale, q_rot,
                                                                tqdm)

        print("-( Write dequantized PC )-------------")
        write3DG_ply(r_pos, r_sh, r_opacity, r_scale, r_rot, True, file_dequantized, tqdm)

    def encoder(self):
        file=branch_selected[0] + "/" + self.condition_selecte + "/" + self.class_selecte+"/"+self.rate_point+"__Bitbream.txt"
        with open(file, "a") as f:
            main = self.tmc13

            cfg_path = os.getcwd()+"/cfg"+str(tmc3_selected)+"/"+branch_selected[0]+"/"+condition_selected[self.condition_selecte]+"/"+self.rate_point
            print("cfg路径为："+cfg_path)
            para_encfg = "-c " + cfg_path + "/encoder.cfg"
            para_decfg = "-c " + cfg_path + "/decoder.cfg"

            para2 = "--uncompressedDataPath=" + self.output+"/"+"quantized.ply"
            para3 = "--compressedStreamPath=" + self.output+"/"+"compress.bin"
            para4 = "--reconstructedDataPath=" + self.output+"/"+"encoder.ply"
            para = "%s %s %s %s %s" % (main, para_encfg, para2, para3, para4)
            r = subprocess.run(para, capture_output=True, text=True)
            print(r.stdout,file=f)
            if not os.path.exists(self.output+"/"+"encoder.ply"):
                print("编码端重建失败")
                print("运行配置为: " + para)


            para4 = "--reconstructedDataPath=" + self.output+"/"+"decoder.ply"
            para = "%s %s %s %s" % (main, para_decfg, para3, para4)
            r = subprocess.run(para, capture_output=True, text=True)
            if not os.path.exists(self.output+"/"+"decoder.ply"):
                print("解码端重建失败")
                print("运行配置为: " + para)
                print(r.stdout)

    def render(self):
            os.rename(self.base+"/point_cloud/iteration_30000/point_cloud.ply",self.base+"/point_cloud/iteration_30000/tem_point_cloud.ply")
            shutil.copy(self.output + "/dequantized.ply", self.base+"/point_cloud/iteration_30000/point_cloud.ply")

            # 切换到目标目录
            os.chdir("gaussiansplatting")
            print(f"切换到目录: {os.getcwd()}")

            # 执行命令：python convert.py -s data
            cmd = [sys.executable, 'render.py', '-m', self.base]
            subprocess.run(cmd, check=True)
            os.chdir("..")

            shutil.move(self.base+"/train/ours_30000/renders",self.output)
            os.rename(self.output+"/renders",self.output+"/"+self.class_selecte)

            os.remove(self.base+"/point_cloud/iteration_30000/point_cloud.ply")
            os.rename(self.base+"/point_cloud/iteration_30000/tem_point_cloud.ply",self.base+"/point_cloud/iteration_30000/point_cloud.ply")

    def metrics(self):
        main="ctc/QMIV.exe"
        gt=File.get_all_file_from_baseCatalog(".png","image/"+self.class_selecte)
        render=File.get_all_file_from_baseCatalog(".png",self.output+"/"+self.class_selecte)
        metric_path=branch_selected[0]+"/"+self.condition_selecte+"/"+self.class_selecte+"/"+self.rate_point+"__metrics.txt"

        with open(metric_path, "w") as f:
            for i in range(len(gt)):
                #测试rgb下PSNR
                para1="-i0 "+gt[i]
                para2="-i1 "+render[i]
                para3="-ml PSNR"

                image = plt.imread(gt[i])
                height, width = image.shape[:2]
                para4="-ps "+str(width)+"x"+str(height)
                para5="-ff PNG"
                para6="-cwa "+"1:1:1:0"
                para7="-cws "+"1:1:1:0"
                para = "%s %s %s %s %s %s %s %s" % (main, para1, para2, para3,para4,para5,para6,para7)

                r = subprocess.run(para, capture_output=True, text=True)
                print(r.stdout,file=f)

               #测试yuv下PSNR,SSIM,IVSSIM

                para3 = "-ml PSNR,SSIM,IVSSIM"

                para6 = "-csm YCbCr_BT709"
                para = "%s %s %s %s %s %s %s" % (main, para1, para2, para3, para4, para5, para6)
                r = subprocess.run(para, capture_output=True, text=True)
                print(r.stdout,file=f)

                #计算LPIPPS
                loss = lpips.LPIPS(net='alex')
                img0 = lpips.im2tensor(lpips.load_image(gt[i]))
                img1 = lpips.im2tensor(lpips.load_image(render[i]))
                d = loss(img0, img1)
                print(">>>> LPIPS:"+str(np.array(d.data.cpu())[0,0,0,0]),file=f)

    def write_to_excel(self):
        # ======================================
        # 写入PSNR
        # ======================================
        metrics_path=File.get_all_file_from_baseCatalog("__metrics.txt",branch_selected[0])
        metrics_data=dict()
        for path in metrics_path:
            metrics_data[path]=self.extract_metrics(path)

        self.sub_write_to_excel(metrics_data, self.output_excel)


        # ======================================
        # 写入编码比特
        # ======================================
        bitstream_files=File.get_all_file_from_baseCatalog("__Bitbream.txt",branch_selected[0])
        for bitstream_file in bitstream_files:

            labels=bitstream_file.split("/")
            increase_row=int(labels[-1][1:3])-1

            parsed_data = self.parse_bitstream(bitstream_file)
            aggregated = self.aggregate_attributes(parsed_data)
            self.sub_bitstream_write_to_excel(aggregated, row=self.Bitstream_start_row+increase_row)


        # 保存为新文件（保留宏）
        self.wb.save(branch_selected[0]+"/"+self.output_excel)
        print(f"数据已成功写入 {self.output_excel}")






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
            data[frame_id] = {
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
    def sub_write_to_excel(self,metrics_data, output_path):

        # 选择工作表（假设目标表名为 "Results"）
        ws = self.wb[Mandatory_new[self.class_selecte][0]]  # 如果表名不同，需修改此处
        columns=self.test_columns if self.tmc else self.anchor_columns

        # 定义数据列的起始位置（根据你的 Excel 模板调整）
        # 示例：假设表头在行1，数据从行2开始
        start_row = self.PSNR_start_row
        # 按图像编号顺序写入数据
        for path in metrics_data.keys():
            labels=path.split("/")
            increase_row=int(labels[-1][1:3])-1
            for view in sorted(metrics_data[path].keys()):
                row = start_row + len(self.rate_points)*view+increase_row# 假设数据按顺序排列
                data = metrics_data[path][view]

                # 写入指标数据（保留原始精度）
                ws[f'{columns["PSNR-RGB"]}{row}'] = data["PSNR-RGB"]
                ws[f'{columns["PSNR-YCbCr"]}{row}'] = data["PSNR-YCbCr"]
                ws[f'{columns["SSIM-YCbCr"]}{row}'] = data["SSIM-YCbCr"]
                ws[f'{columns["IVSSIM"]}{row}'] = data["IVSSIM"]
                ws[f'{columns["LPIPS"]}{row}'] = data["LPIPS"]



# ======================================
# 步骤 3：提取属性的编码bit
# ======================================
    def parse_bitstream(self,file_path):
        """使用正则表达式解析比特流文件"""
        attributes = {}
        pattern = re.compile(r'^(\w+).*bitstream size (\d+) B \((\d+\.\d+) bpp\)')

        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    attr = match.group(1)  # 属性名
                    size = int(match.group(2))  # 字节数
                    bpp = float(match.group(3))  # bpp
                    attributes[attr] = {"size": size, "bpp": bpp}
        return attributes

# ======================================
# 步骤 4：属性聚类
# ======================================
    def aggregate_attributes(self,attrs):
        """聚合属性到指定分类"""
        return {
            "position": attrs.get("positions", {}).get("size", 0),
            "sh0": attrs.get("f_dc_0s", {}).get("size", 0),
            "sh1": attrs.get("f_dc_1s", {}).get("size", 0),
            "sh2": attrs.get("f_dc_2s", {}).get("size", 0),
            "sh3": sum(v["size"] for k, v in attrs.items() if k.startswith("f_rest_")),
            "rotation": sum(v["size"] for k, v in attrs.items() if k.startswith("rot_")),
            "scaling": sum(v["size"] for k, v in attrs.items() if k.startswith("scale_")),
            "opacity": attrs.get("opacitys", {}).get("size", 0),
            "metadata": 0  # 元数据占位符
        }

# ======================================
# 步骤 5：编码比特写入
# ======================================
    def sub_bitstream_write_to_excel(self,data,row=16):
        ws = self.wb[Mandatory_new[self.class_selecte][0]]
        # 列映射关系
        columns = self.test_Bitstream_columns if tmc3_selected else self.anchor_Bitstream_columns

        # 写入数据
        for field, col in columns.items():
            cell = f"{col}{row}"
            ws[cell] = data.get(field, 0) * 8 / 1000

