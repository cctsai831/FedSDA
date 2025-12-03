import sde
import ml_collections
import torch
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import stain_fid
from absl import logging
import builtins
import os
import wandb
import pandas as pd

def main_csv(filename, epoch, loss):
    df = pd.DataFrame({'Epoch':[epoch], 'loss': [loss]})
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def fid_csv(filename, fids_avg, fids_H, fids_E):
    df = pd.DataFrame({'Avg': fids_avg, 'H': fids_H, 'E': fids_E})
    # Check if the file exists. If it does not, write the header.
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def evaluate(config, train_states, device):
    fids_H, fids_E, fids_avg = [], [], []
    ccs = [84098, 223645, 73266, 123238, 41428]
    for cc in range(5):
        config.sample.n_samples = ccs[cc]
        nnet = train_states[0].nnet
        nnet.eval()
        dataset = get_dataset("StainEval", client=cc)
        if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
            logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
            def cfg_nnet(x, timesteps, y):
                _cond = nnet(x, timesteps, y=y)
                _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
                return _cond + config.sample.scale * (_cond - _uncond)
            score_model = sde.ScoreModel(cfg_nnet, pred=config.pred, sde=sde.VPSDECosine())
        else:
            score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDECosine())

        print(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}')

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError

            if config.sample.algorithm == 'euler_maruyama_sde':
                rsde = sde.ReverseSDE(score_model)
                return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=False, **kwargs)
            elif config.sample.algorithm == 'euler_maruyama_ode':
                rsde = sde.ODE(score_model)
                return sde.euler_maruyama(rsde, x_init, config.sample.sample_steps, verbose=False, **kwargs)
            elif config.sample.algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(
                    x_init,
                    steps=config.sample.sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError
        path = os.path.join(config.train.svaes_path, str(cc))
        os.makedirs(path, exist_ok=True)
        utils.sample2dir_2(path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        fid_H, fid_E, fid_avg = stain_fid(path, cc)
        print(f"FID_avg : {fid_avg}, FID_H : {fid_H}, FID_E : {fid_E}")
        fids_H.append(fid_H)
        fids_E.append(fid_E)
        fids_avg.append(fid_avg)
    fid_csv(config.train.fid_path, fids_avg, fids_H, fids_E)
        

def train(config):
    client = 5# config.dataset.cls_number
    print(f'How many client:{client}')
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mini_batch_size = config.train.batch_size

    train_dataset_loaders = []
    for center in range(client):
        dataset = get_dataset(**config.dataset, center=center)
        train_dataset_loaders.append(DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, drop_last=False,
                                        num_workers=8, pin_memory=True, persistent_workers=True))

    train_states = [utils.initialize_train_state(config, device) for _ in range(client)]
    server_train_state = utils.initialize_train_state(config, device)

    def get_data_generator(train_dataset_loader):
        while True:
            for data in train_dataset_loader:
                yield data

    data_generators = [get_data_generator(train_dataset_loader) for train_dataset_loader in train_dataset_loaders]

    def train_step(_batch, score_model, optimizer, train_state):
        _metrics = dict()
        optimizer.zero_grad()
        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch, pred=config.pred)
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred, y=_batch[1])
        else:
            raise NotImplementedError(config.train.mode)
        loss = loss.mean()
        _metrics['loss'] = loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def communication(config, server_train_state, train_states, client_weights, epoch, mean_loss, best_loss):
        server_train_state.nnet.eval()
        server_dict = server_train_state.nnet.state_dict()
        for k in server_dict.keys():
            server_dict[k] = torch.stack([train_states[i].nnet.state_dict()[k].float().mul(client_weights[i]) for i in range(len(train_states))], 0).sum(0)
        server_train_state.nnet.load_state_dict(server_dict)
        
        for train_state in train_states:
            train_state.nnet.load_state_dict(server_train_state.nnet.state_dict())
        
        try:
            state_dict = server_train_state.nnet.module.state_dict()
        except AttributeError:
            state_dict =  server_train_state.nnet.state_dict()
            
        if best_loss > mean_loss:
            best_loss = mean_loss
            save_path = os.path.join(config.train.model_path, "best.pt")
            print(f"Save Best Model : {epoch}")
            torch.save(state_dict, save_path)

        print(f"Save {epoch+1}/{config.train.n_steps} Model : {epoch}")
        save_path = os.path.join(config.train.model_path, f"{epoch}.pt")
        torch.save(state_dict, save_path)
        return best_loss
    
    print(f'Start fitting')

    server_epoch = config.train.n_steps
    client_weights = config.train.client_weights
    best_loss = 10
    for epoch in range(server_epoch):
        loss_ = 0
        for c in range(client):
            for local in range(config.train.local_epoch):
                train_state = train_states[c]
                nnet = train_state.nnet
                optimizer = train_state.optimizer
                lr_scheduler = train_state.lr_scheduler
                score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
                nnet.train()
                batch = tree_map(lambda x: x.to(device), next(data_generators[c]))
                metrics = train_step(batch, score_model, optimizer, train_state)
                loss_ += metrics['loss']
        mean_loss = loss_/client/config.train.local_epoch
        print(f"Sever Epoch:{epoch}, loss:{mean_loss}")
        main_csv(config.train.csvpath, epoch, mean_loss)
        best_loss = communication(config, server_train_state, train_states, client_weights, epoch, mean_loss, best_loss)
        # if epoch%100==0:
        #     evaluate(config, train_states, device)
    print(f'Finish fitting')
    try:
        state_dict = server_train_state.nnet.module.state_dict()
    except AttributeError:
        state_dict =  server_train_state.nnet.state_dict()
    if epoch%100==0:
        save_path = os.path.join(config.train.model_path, f"{epoch}.pt")
        torch.save(state_dict, save_path)
    save_path = os.path.join(config.train.model_path, "last.pt")
    torch.save(state_dict, save_path)


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
