
###--------------------------------------------------
'''
文件夹目录,需要手动粘贴的只有cfg文件夹
.-||
  |--cfg0 (cfg1) 运行anchor时为cfg0
  |--ctc
  |--example
  |--gaussian-splatting
  |--image
  ....
'''


###--------------------------------------------------
#1F 或者 NF
NF="1F"

# 渲染图像的宽、高、视角数、起始帧、帧数、Excel所在行
seq_information_NF = {
    #NF: Forward facing & Object-centric sequences
"bartender_semitracked":[1920,1080,21,0,32,3],
"cinema_semitracked"   :[1920,1080,21,0,32,9],
"breakfast_semitracked":[1920,1080,15,0,32,15],
"breakfast_untracked"  :[1920,1080,15,0,32,21],
"breakdance_untracked" :[1920,1080,33,0,32,27],
"bartender_tracked"    :[1920,1080,21,0,32,33],
"cinema_tracked"       :[1920,1080,21,0,32,39],
"breakfast_tracked"    :[1920,1080,15,0,32,45],
"manwithfruit_tracked" :[3840,2160,24,81,32,51],
}


# 渲染图像的宽、高、视角数、起始帧、帧数、Excel所在行
seq_information_1F = {
    #1F: Object-centric sequences
"bartender_semitracked":[1920,1080,21,0,1,57]     ,
"cinema_semitracked"   :[1920,1080,21,0,1,63]     ,
"breakfast_semitracked":[1920,1080,15,0,1,69]     ,
"breakfast_untracked"  :[1920,1080,15,0,1,75]     ,
"breakdance_untracked" :[1920,1080,33,0,1,81]     ,
"bartender_tracked"    :[1920,1080,21,0,1,87]     ,
"cinema_tracked"       :[1920,1080,21,0,1,93]     ,
"breakfast_tracked"    :[1920,1080,15,0,1,99]     ,
"manwithfruit_tracked" :[3840,2160,24,81,1,105]   ,
"lego_ferrari"         :[4594,5514,128,0,1,111]   ,
"lego_bugatti"         :[3852,2868,132,0,1,117]   ,
"cricket_player"       :[4520,2540,60,0,1,123]    ,
"plant"                :[2954,3968,66,0,1,129]    ,
"solo_tango_female"    :[4534,2542,44,0,1,135]    ,
"solo_tango_male"      :[4528,2544,44,0,1,141]    ,
"tango_duo"            :[4522,2538,44,0,1,147]    ,
"tennis_player"        :[4518,2540,60,0,1,153]    ,
"library"              :[3793,2131,173292,0,1,159],
"flowerdance"          :[2456,2054,64,0,1,165]    ,
"gymnast"              :[2456,2054,64,200,1,171]  ,

}

seq_information=seq_information_NF if NF=="NF" else  seq_information_1F
###--------------------------------------------------


PSNR_columns = {
    "PSNR-RGB": "T",  # PSNR-RGB 列
    "PSNR-YCbCr": "U",  # PSNR-YCbCr 列
    "SSIM-YCbCr": "V",  # SSIM-YCbCr 列
    "MIN_PSNR-RGB":"Y",
    "MIN_PSNR-YUV":"Z",
    "MIN_SSIM-YUV":"AA",
    "MAX_PSNR-RGB":"AB",
    "MAX_PSNR-YUV":"AC",
    "MAX_SSIM-YUV":"AD"
    # "IVSSIM": "I",  # IVSSIM 列
    # "LPIPS": "J"  # LPIPS 列
}

Bitstream_columns = {
    "Total":"K",
    "position": "L",
    "sh0": "M",
    "sh1": "N",
    "sh2": "O",
    "sh3": "P",
    "rotation": "Q",
    "scaling": "R",
    "opacity": "S",
    # "metadata":"AK",
    "T_Enc": "AE",
    "T_Dec": "AH",
    "G_Enc": "AF",
    "G_Dec": "AI",
    "A_Enc": "AG",
    "A_Dec": "AJ",
    "maxRSS_Enc": "AK",
    "maxRSS_Dec": "AL",
    "CustomA": "W",
    "CustomB": "X",
}


