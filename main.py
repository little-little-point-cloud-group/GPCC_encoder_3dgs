import os.path
import shutil

from GS_tools import Gaussian
from my_tools import File
import multiprocessing

def modify_config_file(attr_id, qp, level):
    attrs = ["DC", "AC", "Opacity", "Scale", "Rotation"]
    attr=attrs[attr_id]
    cfgs_path = "./cfg1/octree-raht_" + attr + "_" + str(level) + "_" + str(qp)
    if os.path.exists(cfgs_path):
        shutil.rmtree(cfgs_path)

    specline={
        "DC":"# SH0 (DC) ---------",
        "AC":"# SH1 Y  (AC) ------",
        "Opacity":"# Opacity ----------",
        "Scale":"# Scale ------------",
        "Rotation":"# Rotation ---------",
    }
    shutil.copytree("./cfg1/octree-raht",cfgs_path)
    # 读取配置文件内容
    cfgs=File.get_all_file_from_baseCatalog("encoder.cfg",cfgs_path)
    for cfg_path in cfgs:

        offsets = [0] * (level - 1) + [qp]
        offsets_str = "qpLayerOffsetsLuma: "+",".join(map(str, offsets))
        if attr=="Rotation":
            File.insert_line_above_target(cfg_path,offsets_str,specline[attr])
        else:
            File.insert_line_above_target(cfg_path, offsets_str, specline[attr])
            File.insert_line_above_target(cfg_path, offsets_str, specline[attrs[attr_id+1]])

    g=Gaussian()
    g.branch=["octree-raht_" + attr + "_" + str(level) + "_" + str(qp),]
    os.makedirs("./1F-geo/"+attr+"/level_"+str(level),exist_ok=True)
    g.out_excel=attr+"/level_"+str(level)+"/"+str(qp)+".xlsm"
    g.run()



if __name__ == '__main__':


    attrs=["DC","AC","Opacity","Scale","Rotation"]
    attrs = [ "Scale", "Rotation"]
    qp_range = range(-4, 2, 2)  # -10到10，步长为2
    level_range = range(6, 13, 5)  # 1到13

    thread_pool1 = multiprocessing.Pool(120)
    # 遍历所有参数组合
    for attr_id in range(5):
        for level in level_range:
            for qp in qp_range:
                if qp==0:
                    continue
                #modify_config_file(attr_id, qp, level)
                thread_pool1.apply_async(modify_config_file,args=(attr_id,qp,level))

    thread_pool1.close()  # 关闭进程池入口，不再接受新进程插入
    thread_pool1.join()  # 主进程阻塞，等待进程池中的所有子进程结束，再继续运行主进程
