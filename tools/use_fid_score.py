from fid_score import save_statistics_of_path

path = ["../../dataset/center_1_patches_CL0_RL5_256_1",
        "../../dataset/center_2_patches_CL0_RL5_256_1",
        "../../dataset/center_3_patches_CL0_RL5_256_1",
        "../../dataset/center_4_patches_CL0_RL5_256_1",
        "../../dataset/center_5_patches_CL0_RL5_256_1"]
output_path = "../../dataset/npz/center_all.npz"
save_statistics_of_path(path, output_path)