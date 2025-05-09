I_3DGS_Test_dataset = {
    "Mandatory": {
        "M-NC1": {
            "Bartender": [21, "Forward facing (NC)", 32], # [views，Content type，frames]
        },
        "M-NC2": {
            "Cinema": [21, "Forward facing (NC)", 32],
        },
        "M-NC3": {
            "Breakfast": [15, "Forward facing (NC)", 32],
        },
        "M-NC9": {
            "ManWithFruit": [24, "Object Centric (NC)", 300],
        }
    },
    "Optional": {
        "M-NC4": {
            "Breakdance": [32, "Forward facing (NC)", None],  # Frame count missing in doc
        },
        "M-NC5": {
            "IMT_Human_Centric": ["43-57", "Object Centric (NC)", None],
        },
        "S-NC1": {
            "KITTI-360": [2, "Unbounded scenes", 70000],
        },
        "M-CG1": {
            "Mirror": [15, "Forward facing (CG)", 100],
        },
        "M-CG2": {
            "Garage": [121, "Forward facing (CG)", 96],
        },
        "M-NC6": {
            "Choreo (Dark version)": [20, "Forward facing (NC)", 300],
        },
        "M-NC7": {
            "VRroom1D": [30, "Forward facing (NC)", 300],
        },
        "M-NC8": {
            "HauntedLamp": [12, "Forward facing (NC)", 300],
        },
        "S-NC2": {
            "ToyCat": [1, "Object centric", 361],
        },
        "S-NC3": {
            "ToyCrocodile": [1, "Object centric", 391],
        },
        "S-NC4": {
            "Lighter": [1, "Object centric", 381],
        }
    }
}

A_3DGS_Test_dataset = {
    "Mandatory": {
        "M-NC1": {
            "Bartender": [21, "Forward facing (NC)", 1830],
        },
        "M-NC2": {
            "Cinema": [21, "Forward facing (NC)", 300],
        },
        "M-NC3": {
            "Breakfast": [15, "Forward facing (NC)", 97],
        },
        "M-NC4": {
            "Breakdance": [32, "Forward facing (NC)", None],
        },
        "M-NC5": {
            "IMT_Human_Centric": ["43-57", "Object Centric (NC)", None],
        },
        "M-NC9": {
            "ManWithFruit": [24, "Object Centric (NC)", 300],
        },
        "S-NC1": {
            "KITTI-360": [2, "Unbounded scenes", 70000],
        }
    },
    "Optional": {
        "M-CG1": {
            "Mirror": [15, "Forward facing (CG)", 100],
        },
        "M-CG2": {
            "Garage": [121, "Forward facing (CG)", 96],
        },
        "M-NC6": {
            "Choreo (Dark version)": [20, "Forward facing (NC)", 300],
        },
        "M-NC7": {
            "VRroom1D": [30, "Forward facing (NC)", 300],
        },
        "M-NC8": {
            "HauntedLamp": [12, "Forward facing (NC)", 300],
        },
        "S-NC2": {
            "ToyCat": [1, "Object centric", 361],
        },
        "S-NC3": {
            "ToyCrocodile": [1, "Object centric", 391],
        },
        "S-NC4": {
            "Lighter": [1, "Object centric", 381],
        }
    }
}

A_3DGS_Test_views = {
    "Mandatory": {
        "M-NC1": {
            "Bartender": ["v08,v10,v12", "All remaining views", 50, 65],# [views，Training views，Initial frame，Training frame ]
        },
        "M-NC2": {
            "Cinema": ["v08,v10,v12", "All remaining views", 235, 65],
        },
        "M-NC3": {
            "Breakfast": ["TBD", "All remaining views", "TBD", "TBD"],
        },
        "M-NC4": {
            "Breakdance": ["TBD", "All remaining views", "TBD", "TBD"],
        },
        "M-NC5": {
            "IMT_Human_Centric": ["TBD", "All remaining views", "TBD", "TBD"],
        },
        "M-NC9": {
            "ManWithFruit": ["TBD", "All remaining views", "TBD", "TBD"],
        },
        "S-NC1": {
            "KITTI-360": [
                "2013_05_28_drive_0000_sync image_00:701,703 image_01:701,703",
                "2013_05_28_drive_0000_sync image_00:700,702,704 image_01:700,702,704",
                "--",
                3
            ],
        }
    },
    "Optional": {
        "M-CG1": {
            "Mirror": ["v6,v8", "All of the rest", 0, 65],
        },
        "M-CG2": {
            "Garage": [
                "Select 3x3 from non-training views",
                "Select 6x6 views with 1 interval cameras",
                0,
                65
            ],
        },
        "M-NC6": {
            "Choreo (Dark version)": ["v7,v9,v10,v12", "All remaining views", 30, 65],
        },
        "M-NC7": {
            "VRroom1D": ["v07,v11,v20,v24", "All remaining views", 16, 65],
        },
        "M-NC8": {
            "HauntedLamp": [
                "v0(Top-left corner),v11(Bottom-right corner)",
                "All remaining views",
                0,
                65
            ],
        },
        "S-NC2": {
            "ToyCat": ["img_name=8n (n=0,1,2,...)", "All remaining views", None, None],
        },
        "S-NC3": {
            "ToyCrocodile": ["img_name=8n (n=0,1,2,...)", "All remaining views", None, None],
        },
        "S-NC4": {
            "Lighter": ["img_name=8n (n=0,1,2,...)", "All remaining views", None, None],
        }
    }
}

CTC_condition = ("C1", "C2",)
CTC_class = ("I_3DGS", "A_3DGS",)