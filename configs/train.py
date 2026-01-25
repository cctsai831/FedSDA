import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=2,
        batch_size=16384,
        mode='cond',
        client_weights=[0.1541173776, 0.4098501856, 0.1342667339, 0.2258450543, 0.07592064874],
        svaes_path=f'FedSDA_Save_Samples',
        model_path=f'ckps/FedSDA',
        csvpath=f'logs/FedSDA.csv',
        fid_path=f'scores/FedSDA_FD.csv',
        local_epoch=3
    )

    config.optimizer = d(
        name='adamw',
        lr=0.01,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_transformer',
        img_size_x=2, 
        img_size_y=3,
        patch_size=1, 
        in_chans=1, 
        embed_dim=32,
        num_heads=8, 
        mlp_ratio=4., 
        depth=1,
        num_classes=5
    )

    config.dataset = d(
        name='stain',
        # name='c17wilds',
        path="five_centers",
        # path="all",
        # path="../dataset/c17wilds/stains/",
        cls_number=5
    )

    config.sample = d(
        sample_steps=1000,
        n_samples=100000,
        mini_batch_size=10000,
        algorithm='dpm_solver',
        path='Saves/'
    )

    return config
