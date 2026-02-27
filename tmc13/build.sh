#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供目标文件夹路径作为参数"
    echo "用法: $0 <文件夹路径>"
    echo "示例: $0 HP-RAHT+qpoffset"
    exit 1
fi

# 获取参数
FLODER_NAME="$1"
FLODER="$FLODER_NAME"
ZIP_FILE="./${FLODER_NAME}.zip"

echo "目标文件夹: $FLODER"
echo "ZIP文件: $ZIP_FILE"

# 检查文件夹是否存在
if [ ! -d "$FLODER" ]; then
    echo "文件夹 $FLODER 不存在"
    
    # 检查zip文件是否存在
    if [ -f "$ZIP_FILE" ]; then
        echo "找到ZIP文件，正在解压..."
        
        # 解压zip文件
        unzip -q "$ZIP_FILE" -d "$FLODER_NAME"
    
        
        # 执行构建
        echo "执行构建..."
        cd "$FLODER"
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make

        
    else
        # zip文件也不存在
        echo "错误: 既不存在文件夹 $FLODER，也不存在ZIP文件 $ZIP_FILE"
        echo "请确保以下文件之一存在:"
        echo "  1. $FLODER (目录)"
        echo "  2. $ZIP_FILE (ZIP压缩包)"
        exit 1
    fi
else
    # 文件夹已存在，直接执行构建
    echo "执行构建..."
    cd "$FLODER"
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

fi

# 返回原始目录
cd ..
cd ..

echo "操作完成"