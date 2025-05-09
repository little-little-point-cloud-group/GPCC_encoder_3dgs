步骤1:点云预处理，确定每个属性的量化比特

步骤2:确定每个码率的每个系数QP（JEE6.6中有阐述）

步骤3:编码点云

步骤4:后处理

步骤5:根据视角cameras渲染图像（是利用MPEG的渲染工具，JEE6.2中有阐述）NB:目前使用的是不是MPEG的渲染方法

步骤6:计算渲染出的图像的失真（是预处理前渲染出的图像和后处理之后的图像计算PSNR，JEE6.2中有阐述）

步骤7:将计算出的失真填入excel表格


文件夹目录,需要手动粘贴的只有cfg文件夹
.-||
  |--cfg0 (cfg1) 运行anchor时为cfg0
  |--ctc
  |--example
  |--gaussian-splatting
  |--image
  ....

# ======================================
直接在main.py调用run（）即可，main文件中base文件夹路径分布为
base-||
    | --point_cloud
    | --cameras.json
    | --cfg_args
	| --input.ply

关于其他的路径的配置，请看GS_tools.py

目前的octree-raht表格填入的数据时cinema
