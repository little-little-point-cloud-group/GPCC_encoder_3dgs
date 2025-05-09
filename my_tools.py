
import shutil
import os
import copy


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
                path1 = path1 + "\\" + s[i]
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
        if len(list_in) == 1:
            return list_in[0]
        elif len(list_in) == 0:
            return None

        return list_in  # list=get_all_file_from_baseCatalog("encoder",base)

