import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
from model.evspsegnet import evspsegnet
from utils.stcloss import STCLoss

import torch.optim as optim
import mlflow
import tqdm
from utils.eval import evalute


def setup(seed):
    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)


def create_run_directory(config, seed):
    """Create an immutable per-run checkpoint directory and save its config."""
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'), seed, os.getpid()
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.config, run_dir / 'config.yaml')
    return run_dir, started_at


def save_checkpoint(state_dict, checkpoint_path):
    """Avoid leaving a partial checkpoint if saving is interrupted."""
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def write_run_summary(run_dir, started_at, seed, best_loss, best_iou):
    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': seed,
        'best_loss': best_loss,
        'best_iou': best_iou,
        'best_loss_checkpoint': str(run_dir / 'best_loss_seed{}.pt'.format(seed)),
        'best_iou_checkpoint': (
            str(run_dir / 'best_iou_seed{}.pt'.format(seed))
            if best_iou is not None else None
        ),
        'last_checkpoint': str(run_dir / 'last_seed{}.pt'.format(seed)),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    seed=37
    setup(seed)
    device = "cuda:0"
    run_dir, started_at = create_run_directory(cfg, seed)
    best_loss_path = run_dir / 'best_loss_seed{}.pt'.format(seed)
    best_iou_path = run_dir / 'best_iou_seed{}.pt'.format(seed)
    last_path = run_dir / 'last_seed{}.pt'.format(seed)
    print('run directory:', run_dir)

    net = evspsegnet(cfg).train()
    net.cuda()

    dataset = EvUAV(cfg,mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=train_sampler)

    stc_criterion = STCLoss(k=cfg.k,t=cfg.t,cfg=cfg).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = 1e5
    best_iou = None

    #for val
    val_dataset = EvUAV(cfg, mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,collate_fn=val_dataset.custom_collate)
    evaluter = evalute(cfg)

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name=run_dir.name)
    mlflow.log_params({
        'run_directory': str(run_dir),
        'config_path': str(Path(cfg.config).resolve()),
        'seed': seed,
        'max_events_num': cfg.max_events_num,
    })

    for epoch in range(cfg.epochs):
        pbar = tqdm.tqdm(total=len(train_dataloader), unit="Batch", unit_scale=True,
                         desc="Epoch: {}".format(epoch),position=0,leave=True)

        for ev in train_dataloader:
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()

            preds,voxel = net(x)

            loss = stc_criterion(voxel, p2v_map, preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            with torch.no_grad():
                mlflow.log_metric('loss', loss.item())
                if loss.item()<best_loss:
                    save_checkpoint(net.state_dict(), best_loss_path)
                    best_loss = loss.item()
            torch.cuda.empty_cache()

        scheduler.step()
        save_checkpoint(net.state_dict(), last_path)

        with torch.no_grad():
            if epoch>=40:
                net.eval()
                evaluter.matches = {}
                for sample, ev in enumerate(val_dataloader):
                    x = ev['voxel_ev']
                    label = ev['seg_label'].float().cuda()
                    p2v_map = ev['p2v_map'].long().cuda()

                    preds, voxel = net(x)
                    preds = preds[p2v_map].squeeze().cpu()

                    evaluter.matches[str(sample)] = {}
                    evaluter.matches[str(sample)]['seg_pred'] = preds
                    evaluter.matches[str(sample)]['seg_gt'] = label
                iou = evaluter.evaluate_semantic_segmantation_miou()

                if best_iou is None or iou.item() > best_iou:
                    save_checkpoint(net.state_dict(), best_iou_path)
                    best_iou = iou.item()
                mlflow.log_metric('val_iou', iou.item(), step=epoch)
                net.train()

        write_run_summary(run_dir, started_at, seed, best_loss, best_iou)

    mlflow.end_run()
    print('best loss checkpoint:', best_loss_path)
    if best_iou is not None:
        print('best IoU checkpoint:', best_iou_path)
    print('last checkpoint:', last_path)
