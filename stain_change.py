import os
import glob
import copy
import spams
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
# from stain_mixup.utils import get_stain_matrix, get_concentration, od_to_rgb
########## Stain mixup func. ##########
def od_to_rgb(od):
    return np.minimum(np.exp(-od) * 255, 255).astype(np.uint8)

def rgb_to_od(image):
    return np.maximum(-np.log(np.maximum(image.astype(np.float32), 1.0) / 255), 0)

def get_concentration(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    lambda1: float = 0.01,
):
    # concentration = spams.lasso(
    #     X=np.asfortranarray(rgb_to_od(image).reshape((-1, 3)).T),
    #     D=np.asfortranarray(stain_matrix.T),
    #     mode=2,
    #     lambda1=lambda1,
    #     pos=True,
    #     numThreads=1,
    # ).toarray()
    return spams.lasso(
        X=np.asfortranarray(rgb_to_od(image).reshape((-1, 3)).T),
        D=np.asfortranarray(stain_matrix.T),
        mode=2,
        lambda1=lambda1,
        pos=True,
        numThreads=1,
    ).toarray().T.reshape(*image.shape[:-1], -1)

def stain_mixup(
    image: np.ndarray,
    source_stain_matrix: np.ndarray,
    target_stain_matrix: np.ndarray,
) -> np.ndarray:
    
    # Composite
    return od_to_rgb(get_concentration(image, source_stain_matrix) @ target_stain_matrix)

def get_augmented_image(image_path, source_path, target_path, output_path):
    # augmented_image = stain_mixup(
    #     np.array(Image.open(image_path).convert('RGB')),
    #     np.load(source_path),
    #     np.load(target_path),
    # )
    Image.fromarray(stain_mixup(
        np.array(Image.open(image_path).convert('RGB')),
        np.load(source_path),
        np.load(target_path),
    )).save(output_path)

########## Data path func. ##########
def open_txt(txt):
    with open(txt, "r") as f:
        outlier_paths = f.readlines()
    return [path.strip() for path in outlier_paths]

def get_paths(site=None):
    clients = {1:'Akoya', 2:'KFBio', 3:'Leica', 4:'Olympus', 5:'Philips', 6:'Zeiss'}
    base_image_path = f"/workspace/AGGC_patches/Train/{clients[site]}/patches"
    base_stain_path = f"/workspace/AGGC_stains/{clients[site]}"
    image_paths = glob.glob(os.path.join(base_image_path, "*.png"))
    stain_paths = [path.replace(base_image_path, base_stain_path).replace(".png", ".npy") for path in image_paths]
    return image_paths, stain_paths

def random_choice_stain(stain_list):
    random_choice = random.choice(stain_list)
    stain_list.remove(random_choice)
    return random_choice

def main(client):
    clients = {1:'Akoya', 2:'KFBio', 3:'Leica', 4:'Olympus', 5:'Philips', 6:'Zeiss'}
    base_path_gen_stain = "/workspace/stain-mixup/U-ViT/samples/"
    base_path_output    = f"/workspace/stain-mixup/ACCG_CC_fed/{clients[client]}"
    os.makedirs(base_path_output, exist_ok=True)

    out = open_txt("/workspace/stain-mixup/AGGC_CC_fed_outlier.txt")

    # get paths
    image_paths_lib, stain_paths_lib = get_paths(client)
    assert len(stain_paths_lib)==len(image_paths_lib), "Data not par."

    # get gen stain
    stain_gen_lib = [sorted(glob.glob(os.path.join(base_path_gen_stain, f"ACCG_CC_fed_{i}", "*.npy"))) for i in range(6)]

    # start 
    cnt, idx_ = 0, 0
    for idx, (stain_path, image_path) in tqdm(enumerate(zip(stain_paths_lib, image_paths_lib)), total=len(image_paths_lib)):
        if idx % int(len(stain_paths_lib)/6)==0 and idx_<6:
            cnt+=1
            idx_+=1
            target_stain_paths = copy.copy([path for path in stain_gen_lib[int(cnt-1)] if path not in out])
        
        target_path = random_choice_stain(target_stain_paths)
        if target_stain_paths ==[]:
            target_stain_paths = copy.copy([path for path in stain_gen_lib[int(cnt-1)] if path not in out])
        
        get_augmented_image(image_path, stain_path, target_path, os.path.join(base_path_output, image_path.split("/")[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client', type=int, default=0, help='which client')
    args = parser.parse_args()
    main(args.client)
