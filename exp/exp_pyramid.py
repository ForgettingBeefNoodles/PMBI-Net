"""Isolated, resumable experiment protocol shared by TQNet and TQNet_Pyramid.

Uses the project's data_provider, Adam, LR function and loss. No per-epoch test
access. Resume is transactional at EPOCH BOUNDARIES: an interrupted partial
epoch is replayed from the last committed complete optimizer/RNG state.
"""
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from exp.exp_main import Exp_Main
from utils.tools import adjust_learning_rate

ROOT = Path(__file__).resolve().parents[1]


def add_arguments(parser):
    parser.add_argument('--pyramid_experiment', action='store_true', help='Use matched validation-only selection protocol for baseline')
    parser.add_argument('--pyramid_variant', choices=['v2_full', 'baseline', 'patch_only', 'bottom_only', 'top_only', 'full', 'no_phase', 'no_patch_attention', 'parallel'], default='full')
    parser.add_argument('--pyramid_dim', type=int, default=128)
    parser.add_argument('--pyramid_heads', type=int, default=4)
    parser.add_argument('--pyramid_patch_len', type=int, default=24)
    parser.add_argument('--pyramid_patch_attention', type=int, choices=[0, 1], default=1)
    parser.add_argument('--pyramid_expansion', type=int, default=2)
    parser.add_argument('--pyramid_dropout', type=float, default=.1)
    parser.add_argument('--phase_mode', choices=['original', 'calendar'], default='original')
    parser.add_argument('--pyramid_output', default='./pyramid_runs')
    parser.add_argument('--resume', action='store_true', help='Restore complete last committed epoch, replay any partial epoch')
    parser.add_argument('--pyramid_checkpoint', default='', help='Explicit best.pth for test-only; configuration must match')
    parser.add_argument('--pyramid_skip_test', action='store_true', help='Train/validate only; final test can be invoked later')


def inside_project(value):
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Cross-project path prohibited: {p}')
    return p


def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(args):
    ignore = {'is_training', 'resume', 'pyramid_checkpoint', 'pyramid_skip_test',
              'use_gpu', 'device_ids', 'devices', 'pyramid_output', 'checkpoints'}
    cfg = {k: v for k, v in vars(args).items() if k not in ignore}
    cfg['root_path'] = str(inside_project(args.root_path).relative_to(ROOT))
    cfg['data_sha256'] = file_hash(inside_project(inside_project(args.root_path)/args.data_path))
    protected = ['models/TQNet.py', 'models/TQNet_Pyramid.py', 'exp/exp_pyramid.py',
                 'data_provider/data_loader.py', 'data_provider/data_factory.py', 'utils/tools.py',
                 'exp/exp_main.py', 'run.py']
    cfg['source_sha256'] = {p: file_hash(ROOT/p) for p in protected}
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    variant = 'baseline' if args.model == 'TQNet' else args.pyramid_variant
    dataset = Path(args.data_path).stem
    ident = f'{dataset}_{args.model}_{variant}_r{args.pyramid_dim}_p{args.cycle}_h{args.pred_len}_s{args.random_seed}_{digest}'
    # Keep Windows paths short; full model/variant/config is always in config.json.
    short = f'{dataset[:4]}_{"tq" if args.model == "TQNet" else "pyr"}_{variant[:6]}_r{args.pyramid_dim}_p{args.cycle}_h{args.pred_len}_s{args.random_seed}_{digest}'
    return cfg, short, ident


def atomic_torch(obj, path):
    path = Path(path)
    torch.save(obj, str(path)+'.tmp')
    os.replace(str(path)+'.tmp', path)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class RunLock:
    """OS-released exclusive lock; no stale PID lock after interruption."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, 'a+b')
        try:
            # fstat does not read another process's locked byte on Windows.
            if os.fstat(self.f.fileno()).st_size == 0:
                self.f.write(b'0'); self.f.flush()
            self.f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise RuntimeError('Duplicate run refused: another process holds this experiment lock')
        return self

    def __exit__(self, *args):
        self.f.close()


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, text):
        self.console.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        self.console.flush(); self.file.flush()


def record_failure(folder):
    with open(folder/'FAILED.txt', 'a', encoding='utf-8') as f:
        f.write(time.strftime('\n%Y-%m-%d %H:%M:%S\n')+traceback.format_exc())


def completed_metrics(folder):
    """Do not skip solely because a checkpoint/flag/file name exists."""
    path = folder/'metrics.json'
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding='utf-8'))
    if not all(np.isfinite(m[k]) and m[k] >= 0 for k in ('mse', 'mae')):
        raise ValueError('Invalid existing metrics; refused to overwrite')
    if m['checkpoint_sha256'] != file_hash(folder/'checkpoints/best.pth'):
        raise ValueError('Completed checkpoint hash mismatch')
    for name in ('pred.npy', 'true.npy'):
        arr = np.load(folder/name, mmap_mode='r')
        if len(arr) != m['test_samples'] or arr.size != m['test_elements']:
            raise ValueError('Completed prediction artifact shape mismatch')
    return m


class CalendarDataset(Dataset):
    def __init__(self, base, full_dates, cycle):
        self.base = base
        # Calendar phase is deliberately restricted to verified regular hourly
        # date grids and 24/168 periods. Irregular dates BLOCK this optional mode.
        ns = full_dates.astype('int64').to_numpy()
        diff = np.diff(ns)
        if not np.all(diff == 3600*10**9) or cycle not in (24, 168):
            raise ValueError('calendar mode requires a continuous hourly grid and cycle=24 or168; use original otherwise')
        n = len(full_dates)
        if base.__class__.__name__ == 'Dataset_ETT_hour':
            starts = [0, 12*30*24-base.seq_len, 16*30*24-base.seq_len]
        elif base.__class__.__name__ == 'Dataset_Custom':
            starts = [0, int(n*.7)-base.seq_len, n-int(n*.2)-base.seq_len]
        else:
            raise ValueError('calendar mode only supports verified hourly ETT/Custom datasets')
        self.start = starts[base.set_type]
        self.phases = (full_dates.dt.dayofweek.to_numpy()*24+full_dates.dt.hour.to_numpy()) % cycle

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        *items, _ = self.base[i]
        return (*items, torch.tensor(self.phases[self.start+i+self.base.seq_len]))


class Exp_Pyramid(Exp_Main):
    def _get_data(self, flag):
        dataset, loader = super()._get_data(flag)
        if self.args.phase_mode == 'calendar':
            dates = pd.to_datetime(pd.read_csv(inside_project(self.args.root_path)/self.args.data_path, usecols=['date'])['date'], errors='raise')
            dataset = CalendarDataset(dataset, dates, self.args.cycle)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=flag != 'test',
                                drop_last=flag != 'test', num_workers=0)
        return dataset, loader

    def batch(self, batch):
        x, y, _, _, phase = batch
        x = x.float().to(self.device)
        y = y[:, -self.args.pred_len:, :].float().to(self.device)
        phase = phase.long().to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            out = self.model(x, phase)
            f = -1 if self.args.features == 'MS' else 0
            out, y = out[:, -self.args.pred_len:, f:], y[:, :, f:]
            if out.shape != y.shape:
                raise ValueError(f'Prediction/label mismatch: {out.shape} vs {y.shape}')
            loss = torch.nn.functional.mse_loss(out, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        return out, y, loss

    def validation(self, loader):
        # Match legacy validation: shuffled drop_last=True batches, arithmetic
        # mean of batch MSE. All full batches have equal size.
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                out, y, _ = self.batch(batch)
                losses.append(torch.nn.functional.mse_loss(out.float().cpu(), y.float().cpu()).item())
        if not losses:
            raise ValueError('Empty validation loader')
        return float(np.mean(losses))

    def train_isolated(self, folder, cfg):
        a = self.args
        train_data, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        if getattr(train_data, 'base', train_data).data_x.shape[-1] != a.enc_in:
            raise ValueError('enc_in does not match the dataset')
        if not len(train_loader):
            raise ValueError('Empty training loader')
        optimizer = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
            pct_start=a.pct_start, epochs=a.train_epochs, max_lr=a.learning_rate)
        scaler = torch.amp.GradScaler('cuda', enabled=a.use_amp and self.device.type == 'cuda')
        last = folder/'checkpoints'/'last.pth'
        best = folder/'checkpoints'/'best.pth'
        state = dict(next_epoch=0, best_loss=float('inf'), best_epoch=0, counter=0,
                     early_stop=False, history=[], elapsed_seconds=0., peak_vram_bytes=0)
        best_weights = None
        if last.exists():
            if not a.resume:
                raise RuntimeError('Partial experiment exists. Use --resume to restore complete optimizer/scheduler/AMP/RNG state')
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['config'] != cfg:
                raise ValueError('Resume configuration/source/data signature mismatch')
            self.model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['amp'])
            state, best_weights = saved['progress'], saved['best_weights']
            restore_rng(saved['rng'])
            if best_weights is not None:
                atomic_torch(best_weights, best)
            print(f'RESUME: next epoch {state["next_epoch"]+1}; complete state restored; partial epoch replayed')

        # Persist the actual training-only StandardScaler as an audit artifact.
        base_data = getattr(train_data, 'base', train_data)
        if hasattr(base_data, 'scaler'):
            scaler_path = folder/'training_scaler.npz'
            if not scaler_path.exists():
                np.savez(scaler_path, mean=base_data.scaler.mean_, scale=base_data.scaler.scale_, var=base_data.scaler.var_)

        def commit():
            atomic_torch(dict(config=cfg, model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), amp=scaler.state_dict(), progress=state,
                best_weights=best_weights, rng=rng_state()), last)
            if best_weights is not None:
                atomic_torch(best_weights, best)
            if state['history']:
                with open(folder/'training_log.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(state['history'][0]))
                    writer.writeheader(); writer.writerows(state['history'])

        if not last.exists():
            commit()  # Initial full state also makes interruptions in epoch1 recoverable.
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for epoch in range(state['next_epoch'], a.train_epochs):
                if state['early_stop']:
                    break
                started = time.perf_counter()
                self.model.train()
                losses = []
                actual_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}: nominal LR={a.learning_rate:.9g}, actual optimizer LR={actual_lr:.9g}', flush=True)
                for i, batch in enumerate(train_loader):
                    optimizer.zero_grad(set_to_none=True)
                    _, _, loss = self.batch(batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                    losses.append(loss.item())
                    if a.lradj == 'TST':
                        adjust_learning_rate(optimizer, scheduler, epoch+1, a, printout=False)
                        scheduler.step()
                    if (i+1) % 100 == 0:
                        print(f'  step {i+1}/{len(train_loader)} loss={loss.item():.8f}', flush=True)
                val_loss = self.validation(val_loader)
                # Identical delta=0 early stopping tie policy, NumPy2-safe infinity.
                if val_loss <= state['best_loss']:
                    state.update(best_loss=val_loss, best_epoch=epoch+1, counter=0)
                    best_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    state['counter'] += 1
                state['early_stop'] = state['counter'] >= a.patience
                # Legacy does not apply epoch LR update after an early-stop break.
                if not state['early_stop'] and a.lradj != 'TST':
                    adjust_learning_rate(optimizer, scheduler, epoch+1, a)
                row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_mse=val_loss,
                    best_epoch=state['best_epoch'], nominal_lr=a.learning_rate, actual_lr=actual_lr,
                    next_lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
                state['history'].append(row)
                state['next_epoch'] = epoch+1
                state['elapsed_seconds'] += row['seconds']
                if self.device.type == 'cuda':
                    state['peak_vram_bytes'] = max(state['peak_vram_bytes'], torch.cuda.max_memory_allocated(self.device))
                commit()
                print(f'Epoch {epoch+1} | train={row["train_loss"]:.8f} val={val_loss:.8f} best={state["best_epoch"]} patience={state["counter"]}/{a.patience}', flush=True)
            if best_weights is None:
                raise RuntimeError('No valid best checkpoint')
            atomic_json(dict(best_epoch=state['best_epoch'], best_val_mse=state['best_loss'],
                epochs=state['next_epoch'], train_seconds=state['elapsed_seconds'],
                peak_vram_bytes=state['peak_vram_bytes']), folder/'TRAINED.json')
        except BaseException:
            record_failure(folder)
            raise

    def evaluate_best(self, folder):
        metrics_path = folder/'metrics.json'
        existing = completed_metrics(folder)
        if existing is not None:
            print('SKIP: completed test metrics already exist (no overwrite).')
            if not (folder/'DONE.flag').exists():
                (folder/'DONE.flag').write_text('Verified completed checkpoint, metrics and arrays.\n', encoding='utf-8')
            return existing
        best = folder/'checkpoints'/'best.pth'
        if self.args.pyramid_checkpoint and inside_project(self.args.pyramid_checkpoint) != best:
            raise ValueError('Explicit checkpoint must be this exact configuration-matched best.pth')
        self.model.load_state_dict(torch.load(best, map_location=self.device, weights_only=True), strict=True)
        dataset, loader = self._get_data('test')  # FIRST test dataset access is here.
        if loader.drop_last or getattr(loader.sampler, '__class__').__name__ != 'SequentialSampler':
            raise ValueError('Formal test requires all samples in their original order')
        # Memory-mapped arrays avoid retaining multi-GB Traffic forecasts in RAM.
        n = len(dataset)
        channels = 1 if self.args.features in ('MS', 'S') else self.args.enc_in
        shape = (n, self.args.pred_len, channels)
        pred_path, true_path = folder/'pred.npy', folder/'true.npy'
        ptmp, ttmp = folder/'pred.tmp.npy', folder/'true.tmp.npy'
        preds = np.lib.format.open_memmap(ptmp, mode='w+', dtype='float32', shape=shape)
        trues = np.lib.format.open_memmap(ttmp, mode='w+', dtype='float32', shape=shape)
        self.model.eval()
        offset = count = 0
        sum_sq = sum_abs = 0.
        started = time.perf_counter()
        with torch.inference_mode():
            for batch in loader:
                out, y, _ = self.batch(batch)
                p, t = out.float().cpu().numpy(), y.float().cpu().numpy()
                size = len(p)
                preds[offset:offset+size], trues[offset:offset+size] = p, t
                diff = p.astype(np.float64)-t.astype(np.float64)
                sum_sq += np.square(diff).sum(); sum_abs += np.abs(diff).sum(); count += diff.size
                offset += size
        if offset != n:
            raise ValueError('Test sample count mismatch')
        preds.flush(); trues.flush(); del preds, trues
        os.replace(ptmp, pred_path); os.replace(ttmp, true_path)
        metrics = dict(mse=float(sum_sq/count), mae=float(sum_abs/count), test_samples=n,
            checkpoint_sha256=file_hash(best),
            test_elements=count, inference_seconds=time.perf_counter()-started,
            params=sum(p.numel() for p in self.model.parameters()),
            dataset=Path(self.args.data_path).stem, pred_len=self.args.pred_len,
            seed=self.args.random_seed, model=self.args.model, variant=self.args.pyramid_variant,
            protocol='standardized space; original split/scaler; test drop_last=False; float64 streaming reduction')
        metrics.update(json.loads((folder/'TRAINED.json').read_text(encoding='utf-8')))
        atomic_json(metrics, metrics_path)
        (folder/'DONE.flag').write_text('Best validation checkpoint tested successfully.\n', encoding='utf-8')
        print(f'mse:{metrics["mse"]}, mae:{metrics["mae"]}', flush=True)
        return metrics


def summarize(output):
    rows = []
    for path in sorted(Path(output).glob('*/metrics.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        m['run_directory'] = str(path.parent.relative_to(ROOT))
        rows.append(m)
    if rows:
        pd.DataFrame(rows).sort_values(['dataset', 'pred_len', 'model', 'variant']).to_csv(Path(output)/'summary.csv', index=False)


def run_experiment(args):
    if args.model not in ('TQNet', 'TQNet_Pyramid'):
        raise ValueError('Pyramid protocol only accepts TQNet or TQNet_Pyramid')
    if args.num_workers != 0 or args.itr != 1 or args.use_multi_gpu or args.gpu != 0:
        raise ValueError('This reproducible single-GPU protocol requires workers=0, itr=1, GPU0')
    if not args.use_gpu:
        raise RuntimeError('Formal pyramid experiments require CUDA GPU0; run synthetic CPU checks with check_pyramid.py')
    if args.do_predict or args.output_attention:
        raise ValueError('Use formal forecast/test output; decoder attention and Dataset_Pred are not supported here')
    if args.model == 'TQNet':
        args.pyramid_variant = 'baseline'
    inside_project(args.root_path)
    cfg, ident, full_ident = signature(args)
    output = inside_project(args.pyramid_output)
    folder = output/ident
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'checkpoints').mkdir(exist_ok=True)
    with RunLock(folder/'run.lock'):
        if (folder/'config.json').exists():
            old = json.loads((folder/'config.json').read_text(encoding='utf-8'))
            if old['signature_config'] != cfg:
                raise ValueError('Existing output configuration mismatch; refused overwrite')
        else:
            atomic_json(dict(args=vars(args), signature_config=cfg, experiment_id=full_ident,
                command_line=sys.argv, resume_policy='complete epoch boundary; partial epoch replay',
                selection='validation batch-mean MSE only; no per-epoch test access'), folder/'config.json')
        with open(folder/'train.log', 'a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f'Experiment: {full_ident}\nOutput: {folder}\nCommand: {sys.argv}')
                if completed_metrics(folder) is not None:
                    if not (folder/'DONE.flag').exists():
                        (folder/'DONE.flag').write_text('Verified complete result.\n', encoding='utf-8')
                    print('SKIP: exact completed configuration.'); summarize(output); return
                try:
                    exp = Exp_Pyramid(args)
                    if args.is_training and not (folder/'TRAINED.json').exists():
                        exp.train_isolated(folder, cfg)
                    if not args.pyramid_skip_test:
                        exp.evaluate_best(folder)
                    summarize(output)
                except BaseException:
                    record_failure(folder)
                    raise
