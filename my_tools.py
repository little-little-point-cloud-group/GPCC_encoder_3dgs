
import shutil
import os
import copy
import subprocess
import re

class File:
    def copy_file(ori_path, copy_path):
        is_have_ori = os.path.exists(ori_path)

        if not is_have_ori:
            print("can not find the ori_file that will be copied")
        is_have_target = os.path.exists(copy_path)
        if not is_have_target:
            shutil.copy(ori_path, copy_path)

    def generate_file(target_path=" ", ori_path=" "):
        def makedir(path):
            if os.path.exists(path):
                print(path, "has exist")
                return False
            s = path.split("/")
            path1 = s[0]
            for i in range(1, len(s) - 1):
                path1 = path1 + "/" + s[i]
            if not os.path.exists(path1):
                makedir(path1)
            os.makedirs(path)
            return True

        if ori_path == " ":
            makedir(target_path)
        else:
            File.copy_file(ori_path,
                           target_path)  # generate_file(base【文件名如.txt】,ori_path=r'D:\vs\attrs.yaml') or generate_file(base【可以使文件夹】)

    def get_all_file_from_baseCatalog(name, ori_path):
        def add_AllPathFromBase_into_list(name, ori_path, list_in):
            list_out = copy.deepcopy(list_in)
            list_path = os.listdir(ori_path)
            for file in list_path:
                if file.find("~&") == 0:
                    continue
                path = ori_path + "/" + file
                if os.path.isdir(path):
                    list_tem = add_AllPathFromBase_into_list(name, path, [])
                    n = len(list_tem)
                    for i in range(n):
                        list_out.append(list_tem[i])
                else:
                    if file.find(name) >= 0:
                        list_out.append(path)
            return list_out

        list_in = []
        list_in = add_AllPathFromBase_into_list(name, ori_path, list_in)
        if len(list_in) == 0:
            return None

        return list_in  # list=get_all_file_from_baseCatalog("encoder",base)


class YUVPlayer:
    class VideoProcessor:
        def __init__(self, filename):
            self.filename = filename
            self.pix_fmt = None
            self._detect_resolution()
            self._detect_pix_fmt()

        def _detect_resolution(self):

            # 匹配 数字x数字 格式（如 3840x2160）
            match = re.search(r'(\d+)x(\d+)', self.filename)
            if match:
                self.width = int(match.group(1))
                self.height = int(match.group(2))
            else:
                # 若未找到则尝试默认值或报错
                self.width = 1920
                self.height = 1080
                print(f"警告：文件名 {self.filename} 中未检测到分辨率，使用默认 1920x1080")

        def get_pix_fmts(self):
            try:
                cmd = [r'example\ffmpeg\ffplay.exe', '-hide_banner', '-pix_fmts']
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = proc.stdout
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Error获取像素格式列表: {e}")
                return []

            pix_fmts = []
            start_parsing = False

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('Pixel formats:'):
                    start_parsing = True
                    continue
                if not start_parsing or not line:
                    continue

                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    fmt_name = parts[1]
                    pix_fmts.append(fmt_name)

            # 去重并按名称长度降序排列（优先匹配更长名称）
            pix_fmts = list({fmt: None for fmt in pix_fmts}.keys())  # 去重
            pix_fmts.sort(key=lambda x: -len(x))
            return pix_fmts

        def _detect_pix_fmt(self):

            pix_fmts = self.get_pix_fmts()
            if not pix_fmts:
                print("警告：无法获取像素格式列表，使用默认值yuv420p")
                self.pix_fmt = 'rgb48le'
                return

            # 优先检查完整单词匹配（使用正则表达式）
            filename_lower = self.filename.lower()
            for fmt in pix_fmts:
                pattern = r'\b' + re.escape(fmt.lower()) + r'\b'
                if re.search(pattern, filename_lower):
                    self.pix_fmt = fmt
                    return

            # 如果未找到单词匹配，尝试子字符串匹配
            for fmt in pix_fmts:
                if fmt.lower() in filename_lower:
                    self.pix_fmt = fmt
                    return

            # 最终未找到则使用默认值
            self.pix_fmt = 'rgb24'
            print(f"警告：未在文件名 {self.filename} 中发现像素格式，已设为默认rgb24")

    def __init__(self, path):
        self.rawvideo = path
        self.ffmpeg_play = r'example\ffmpeg\ffplay.exe'  # FFmpeg可执行路径
        self.ffmpeg_path = r'example\ffmpeg\ffmpeg.exe'
        self.yuv_file = path  # YUV文件路径
        h_video = self.VideoProcessor(path)
        self.width = h_video.width  # 视频宽度
        self.height = h_video.height  # 视频高度
        self.pix_fmt = h_video.pix_fmt  # YUV像素格式
        self.framerate = 2  # 帧率

    def play(self):
        # 组装FFmpeg命令参数
        main = self.ffmpeg_play
        input_params = f" -video_size {self.width}x{self.height} -pixel_format {self.pix_fmt} -framerate {self.framerate}"
        input_file = f"-i \"{self.yuv_file}\""

        # 拼接完整命令
        command = f"{main} {input_params} {input_file}"

        # 执行命令
        os.system(command)

    def save_frames(self, start_frame: int, num_frames=-1, output_dir: str = "out_image"):
        """
        使用 FFmpeg 截取视频中的图片帧。

        :param start_frame: 截取开始的帧数
        :param num_frames: 截取的帧数
        :param output_dir: 图片输出目录
        """
        # 如果没有指定输出目录，则使用当前目录
        if not output_dir:
            output_dir = os.getcwd()

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        v = self.rawvideo.split('\\')[-1].split('_')[0]
        # 设置输出图像文件的路径（例如 frame_0001.png）
        output_path = os.path.join(output_dir, v + "_frame_%04d.jpg")

        # 构建 FFmpeg 命令
        if num_frames==-1:
            command = [
            self.ffmpeg_path,  # ffmpeg 的路径
            '-video_size', f'{self.width}x{self.height}',  # 视频分辨率
            '-pixel_format', self.pix_fmt,  # 像素格式
            '-framerate', str(self.framerate),  # 帧率
            '-i', self.yuv_file,  # 输入 YUV 文件
            #'-vf', f'select=between(n\,{start_frame}\,{start_frame + num_frames - 1}),setpts=N/FRAME_RATE/TB',
            # 选择要提取的帧
            '-f', 'image2',  # 输出格式为图像序列
            output_path  # 输出文件路径
        ]
        else:
            command = [
                self.ffmpeg_path,  # ffmpeg 的路径
                '-video_size', f'{self.width}x{self.height}',  # 视频分辨率
                '-pixel_format', self.pix_fmt,  # 像素格式
                '-framerate', str(self.framerate),  # 帧率
                '-i', self.yuv_file,  # 输入 YUV 文件
                # '-vf', f'select=between(n\,{start_frame}\,{start_frame + num_frames - 1}),setpts=N/FRAME_RATE/TB',
                # 选择要提取的帧
                '-f', 'image2',  # 输出格式为图像序列
                '-vframes', str(num_frames),  # 提取的帧数
                output_path  # 输出文件路径
            ]

        try:
            # 执行 FFmpeg 命令
            subprocess.run(command,capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while extracting frames: {e}")


if __name__ == '__main__':
    path=r"C:\Users\31046\Desktop\GPCC_encoder_3dgs\octree-raht\lossless-geom-lossy-attrs\m71763_breakfast_stable\track\r01\dec\frame_gpu_img_dec_1920x1080_8b_i444.rgb"
    a=YUVPlayer(path)
    a.play()
