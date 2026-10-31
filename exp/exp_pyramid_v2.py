"""Portable orchestration; the source Exp_Pyramid training/evaluation is retained."""
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from exp.exp_pyramid import (Exp_Pyramid, ROOT, inside_project, file_hash, RunLock,
    Tee, atomic_json, completed_metrics, record_failure, summarize)

def add_arguments(parser):
    import argparse
    parser.add_argument('--pyramid_v2_experiment',action='store_true',default=argparse.SUPPRESS)
    parser.add_argument('--pyramid_v2_variant',choices=['v2_full'],default='v2_full')
    parser.add_argument('--pyramid_v2_output',default='./runs/')

def check_npz(args):
    path=inside_project(inside_project(args.root_path)/args.data_path)
    if not path.is_file(): raise FileNotFoundError(f'Missing project-local dataset: {path}')
    with np.load(path,allow_pickle=False) as archive:
        raw=archive['data']
        if raw.ndim!=3 or raw.shape[2]<1 or raw.shape[1]!=args.enc_in:
            raise ValueError(f'Unexpected PEMS data shape: {raw.shape}')
        if not np.isfinite(raw[:,:,0]).all(): raise ValueError('Nonfinite PEMS flow')
        return dict(shape=list(raw.shape),flow_shape=list(raw[:,:,0].shape))

class Exp_Pyramid_V2(Exp_Pyramid):
    def _get_data(self,flag):
        if self.args.data=='PEMS':
            if self.args.phase_mode!='original' or self.args.features!='M':
                raise ValueError('PEMS requires original absolute-row phases and M task')
            check_npz(self.args)
        return super()._get_data(flag)

def configure(args):
    if args.model not in ('PMBI_Net','TQNet_Pyramid_V2') or args.pyramid_v2_variant!='v2_full':
        raise ValueError('Only the frozen PMBI-Net V2 Full is exposed')
    args.pyramid_variant=args.pyramid_v2_variant
    args.pyramid_output=args.pyramid_v2_output
    return args

def signature_v2(args):
    configure(args)
    ignored={'is_training','resume','pyramid_checkpoint','pyramid_skip_test','use_gpu',
             'devices','device_ids','checkpoints','pyramid_output','pyramid_v2_output',
             'config','exp_key','check_config'}
    cfg={k:v for k,v in vars(args).items() if k not in ignored}
    data=inside_project(inside_project(args.root_path)/args.data_path)
    if not data.is_file(): raise FileNotFoundError(f'Missing local data: {data}')
    cfg['root_path']=inside_project(args.root_path).relative_to(ROOT).as_posix()
    cfg['data_sha256']=file_hash(data)
    sources=['models/PMBI_Net.py','utils/pmbi_helpers.py','exp/checkpoint_compat.py','models/TQNet.py','models/TQNet_Pyramid.py','models/TQNet_Pyramid_V2.py',
             'exp/exp_main.py','exp/exp_basic.py','exp/exp_pyramid.py','exp/exp_pyramid_v2.py',
             'run.py','data_provider/data_loader.py','data_provider/data_factory.py','utils/tools.py']
    cfg['source_sha256']={p:file_hash(ROOT/p) for p in sources}
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    identity=f'{Path(args.data_path).stem}_{args.model}_v2_full_r{args.pyramid_dim}_p{args.cycle}_h{args.pred_len}_s{args.random_seed}_{digest}'
    return cfg,identity,identity

def run_experiment_v2(args):
    configure(args)
    if args.num_workers!=0 or args.itr!=1 or args.use_multi_gpu or args.gpu!=0:
        raise ValueError('Use workers0, itr1 and single GPU0 (or CPU fallback)')
    if args.do_predict or args.output_attention:
        raise ValueError('Use ordinary forecast output')
    if args.data=='PEMS': check_npz(args)
    if args.pyramid_checkpoint:
        if args.is_training:
            raise ValueError('Explicit legacy checkpoint is evaluation-only; no training allowed')
        from exp.checkpoint_compat import evaluate_legacy_checkpoint
        return evaluate_legacy_checkpoint(args)
    cfg,short,identity=signature_v2(args)
    output=inside_project(args.pyramid_v2_output);folder=output/short
    # Test-only must never construct a random predictor as a fallback.
    if not args.is_training and not (folder/'checkpoints/best.pth').is_file():
        raise FileNotFoundError(f'No best checkpoint for this configuration: {folder}/checkpoints/best.pth; no training started')
    folder.mkdir(parents=True,exist_ok=True);(folder/'checkpoints').mkdir(exist_ok=True)
    with RunLock(folder/'run.lock'):
        config_path=folder/'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text(encoding='utf-8'))['signature_config']!=cfg:
                raise ValueError('Existing output config mismatch; overwrite blocked')
        else:
            atomic_json(dict(args=vars(args),signature_config=cfg,experiment_id=identity,
                command_line=sys.argv,resume_policy='complete epoch state; replay partial epoch',
                selection='validation MSE only; no epoch test access'),config_path)
        with open(folder/'train.log','a',encoding='utf-8',buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                print(f'PMBI-Net experiment: {identity}\nOutput: {folder}')
                try:
                    if completed_metrics(folder) is not None:
                        print('SKIP: matching completed checkpoint and result artifacts verified')
                    else:
                        exp=Exp_Pyramid_V2(args)
                        if args.is_training and not (folder/'TRAINED.json').exists(): exp.train_isolated(folder,cfg)
                        if not args.pyramid_skip_test: exp.evaluate_best(folder)
                    summarize(output)
                except BaseException:
                    record_failure(folder)
                    raise
