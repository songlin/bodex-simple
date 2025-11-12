import numpy as np
import transforms3d
from glob import glob
import os
import trimesh
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import random
from curobo.util.logger import log_warn
from curobo.util_file import load_json, load_scene_cfg, join_path, get_assets_path

import re
GraspNet_1B_Object_Names = {
    0: "cracker box",
    1: "sugar box",
    2: "tomato soup can",
    3: "mustard bottle",
    4: "potted meat can",
    5: "banana",
    6: "bowl",
    7: "mug", # "red mug", # 
    8: "power drill",
    9: "scissors",
    10: "chips can", # "red chips can", #
    11: "strawberry", 
    12: "apple",
    13: "lemon",
    14: "peach",
    15: "pear",
    16: "orange",
    17: "plum",
    18: "knife", 
    19: "blue screwdriver", #
    20: "red screwdriver", #
    21: "racquetball", 
    22: "blue cup", #
    23: "yellow cup", #
    24: "airplane", # "toy airplane", 
    25: "toy gun",  # 
    26: "blue toy part", # workpiece
    27: "metal screw", # 
    28: "yellow propeller", # "yellow propeller", # 
    29: "blue toy part a", #
    30: "blue toy part b", #
    31: "yellow toy part", # 
    32: "padlock",
    33: "toy dragon", # 
    34: "small green bottle", # 
    35: "cleansing foam",
    36: "dabao wash soup",
    37: "mouth rinse",
    38: "dabao sod",
    39: "soap box",
    40: "kispa cleanser",
    41: "darlie toothpaste",
    42: "men oil control",
    43: "marker",
    44: "hosjam toothpaste",
    45: "pitcher cap",
    46: "green dish",
    47: "white mouse",
    48: "toy model", # 
    49: "toy deer", # 
    50: "toy zebra", # 
    51: "toy large elephant", # 
    52: "toy rhinocero", #
    53: "toy small elephant", #
    54: "toy monkey", #
    55: "toy giraffe", #
    56: "toy gorilla", #
    57: "yellow snack box", #
    58: "toothpaste box", #
    59: "soap", 
    60: "mouse", 
    61: "dabao facewash", 
    62: "pantene facewash", # "pantene facewash", #
    63: "head shoulders supreme",
    64: "thera med",
    65: "dove", 
    66: "head shoulder care",
    67: "toy lion", # 
    68: "coconut juice box", 
    69: "toy hippo", # 
    70: "tape",
    71: "rubiks cube", 
    72: "peeler cover",
    73: "peeler",
    74: "ice cube mould"
}


class GraspConfigDataset(Dataset):
    def __init__(self, type, template_path, start, end):
        assert type == "grasp"
        template_path = join_path(get_assets_path(), template_path)
        self.grasp_path_lst = np.random.permutation(sorted(glob(template_path, recursive=True)))[
            start:end
        ]
        log_warn(
            f"From {template_path} get {len(self.grasp_path_lst)} grasps. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.grasp_path_lst)

    def __getitem__(self, index):
        full_path = self.grasp_path_lst[index]
        cfg = np.load(full_path, allow_pickle=True).item()
        
        scene_cfg = load_scene_cfg(cfg["scene_path"][0])
        for k, v in cfg.items():
            cfg[k] = v[0]
        cfg["save_prefix"] = scene_cfg["scene_id"] + "_"
        
        match = re.search(r'_mogen(\d+)', str(full_path))
        if not match:
            raise ValueError
        fidx = match.group(1)
        cfg["file_index"] = fidx
        return cfg


def scenecfg2worldcfg(scene_cfg):
    world_cfg = {}
    for obj_name, obj_cfg in scene_cfg["scene"].items():
        if obj_cfg["type"] == "rigid_object":
            if "mesh" not in world_cfg:
                world_cfg["mesh"] = {}
            world_cfg["mesh"][scene_cfg["scene_id"] + obj_name] = {
                "scale": obj_cfg["scale"],
                "pose": obj_cfg["pose"],
                "file_path": obj_cfg["file_path"],
                "urdf_path": obj_cfg["urdf_path"],
            }
        elif obj_cfg["type"] == "plane":
            if "cuboid" not in world_cfg:
                world_cfg["cuboid"] = {}
            assert obj_cfg["pose"][3] == 1
            world_cfg["cuboid"]["table"] = {
                "dims": [5.0, 5.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }
        else:
            raise NotImplementedError("Unsupported object type")
    return world_cfg


class WorldConfigDataset(Dataset):

    def __init__(self, type, template_path, start, end):
        assert type == "scene_cfg"
        scene_cfg_path = join_path(get_assets_path(), template_path)
        self.scene_path_lst = np.random.permutation(sorted(glob(scene_cfg_path)))[start:end]
        log_warn(
            f"From {scene_cfg_path} get {len(self.scene_path_lst)} scene cfgs. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.scene_path_lst)

    def __getitem__(self, index):
        scene_path = self.scene_path_lst[index]
        scene_cfg = load_scene_cfg(scene_path)
        scene_id = scene_cfg["scene_id"]

        obj_name = scene_cfg["task"]["obj_name"]
        obj_cfg = scene_cfg["scene"][obj_name]
        obj_scale = obj_cfg["scale"]
        obj_pose = obj_cfg["pose"]

        json_data = load_json(obj_cfg["info_path"])
        obj_rot = transforms3d.quaternions.quat2mat(obj_pose[3:])
        gravity_center = obj_pose[:3] + obj_rot @ json_data["gravity_center"] * obj_scale
        obb_length = np.linalg.norm(obj_scale * json_data["obb"]) / 2

        return {
            "scene_path": scene_path,
            "world_cfg": scenecfg2worldcfg(scene_cfg),
            "manip_name": scene_id + obj_name,
            "obj_gravity_center": gravity_center,
            "obj_obb_length": obb_length,
            "save_prefix": f"{scene_id}_",
        }

class GraspNetConfigDataset(Dataset):
    def __init__(self, type, template_path, start, end):
        assert type == "graspnet"

        self.asset_path=os.path.join(os.getcwd(), template_path)
        return

    def __len__(self):
        return len(GraspNet_1B_Object_Names)

    def __getitem__(self, index):
        obj_name = list(GraspNet_1B_Object_Names.values())[index]
        obj_name = obj_name.replace(' ', '_')
        obj_scale = np.array([1.0, 1.0, 1.0])
        #FIXME only random selct one stable pose
        obj_stable_poses=np.load(os.path.join(self.asset_path, f"stable/{index}_stable.npy"),allow_pickle=True)
        obj_stable_pose= random.choice(obj_stable_poses)
        
        obj_file_path=os.path.join(self.asset_path, f"models/{index:03d}/coacd_0.05/coacd_merge.obj")

        #compute gravity center and obb length
        mesh = trimesh.load(obj_file_path, process=False)
        v = mesh.vertices
        gravity_center = np.array(v.mean(axis=0))
        obb = v.max(axis=0) - v.min(axis=0)
        obb_length = np.linalg.norm(obb) / 2

        obj_rot = transforms3d.quaternions.quat2mat(obj_stable_pose[3:])
        gravity_center = obj_stable_pose[:3] + obj_rot @ gravity_center * obj_scale

        #world_cfg
        world_cfg={}
        world_cfg["mesh"]={}
        world_cfg["mesh"][obj_name]={
            "scale": obj_scale,
            "pose": obj_stable_pose,
            "file_path": obj_file_path,
            "urdf_path": obj_file_path.replace("_merge.obj", ".urdf"),
        }

    


        return {
            "scene_path": f"/{obj_name}",
            "world_cfg": world_cfg,
            "manip_name": obj_name,
            "obj_gravity_center": gravity_center,
            "obj_obb_length": obb_length,
            "save_prefix": f"{obj_name}/{obj_name}_",
        }

def _world_config_collate_fn(list_data):
    if "world_cfg" in list_data[0]:
        world_cfg_lst = [i.pop("world_cfg") for i in list_data]
    else:
        world_cfg_lst = None
    ret_data = default_collate(list_data)
    if world_cfg_lst is not None:
        ret_data["world_cfg"] = world_cfg_lst
    return ret_data


def get_world_config_dataloader(configs, batch_size):
    if configs["type"] == "scene_cfg":
        dataset = WorldConfigDataset(**configs)
    elif configs["type"] == "grasp":
        dataset = GraspConfigDataset(**configs)
    elif configs["type"] == "graspnet":
        dataset = GraspNetConfigDataset(**configs)
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_world_config_collate_fn
    )
    return dataloader
