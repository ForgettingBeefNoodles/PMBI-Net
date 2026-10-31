"""Explicit, provenance-checked legacy best-checkpoint evaluation; never training.

The untouched original evaluator runs on an isolated copy of the best checkpoint.
Original checkpoints/configs/metrics are never moved, edited, or relabeled.
"""
import contextlib
import copy
import json
import shutil
import sys
import numpy as np
from exp.exp_pyramid import (ROOT, inside_project, file_hash, atomic_json, RunLock,
    Tee, completed_metrics, record_failure)


def resolve_legacy(args):
    checkpoint=inside_project(args.pyramid_checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(f'Legacy best checkpoint missing: {checkpoint}')
    digest=file_hash(checkpoint)
    # Completed native run: use its own config/metrics. Imported reference:
    # resolve by exact weight hash in the preserved formal provenance manifest.
    candidates=[checkpoint.parent.parent,checkpoint.parent]
    found=None
    for parent in candidates:
        if (parent/'config.json').is_file() and (parent/'metrics.json').is_file():
            cfg=json.loads((parent/'config.json').read_text(encoding='utf-8'))
            metrics=json.loads((parent/'metrics.json').read_text(encoding='utf-8'))
            if metrics['checkpoint_sha256']!=digest: raise ValueError('Legacy checkpoint/metrics hash mismatch')
            found=(cfg['args'],metrics,cfg['signature_config']['data_sha256'],parent)
            break
    if found is None:
        records=json.loads((ROOT/'provenance/formal_configs.json').read_text(encoding='utf-8'))
        matches=[v for v in records.values() if v['checkpoint_sha256']==digest]
        if len(matches)!=1: raise ValueError('No unique trusted config for the explicit checkpoint')
        rec=matches[0]
        found=(rec['original_args'],rec['metrics'],rec['data_sha256'],checkpoint.parent)
    original,metrics,data_sha,parent=found
    if original['model'] not in ('TQNet_Pyramid_V2','PMBI_Net') or original['pyramid_v2_variant']!='v2_full':
        raise ValueError('Checkpoint is not formal PMBI-Net Full')
    ignored={'model','is_training','root_path','checkpoints','pyramid_output','pyramid_v2_output',
             'pyramid_checkpoint','resume','use_gpu','devices','device_ids','config','exp_key','check_config',
             'pyramid_skip_test'}
    differences={k:(v,getattr(args,k,None)) for k,v in original.items()
                 if k not in ignored and getattr(args,k,None)!=v}
    if differences: raise ValueError('Legacy configuration mismatch: '+str(differences))
    data=inside_project(inside_project(args.root_path)/args.data_path)
    if file_hash(data)!=data_sha: raise ValueError('Data differs from legacy experiment')
    trained=parent/'TRAINED.json';scaler=parent/'training_scaler.npz'
    if not trained.is_file() or not scaler.is_file():
        raise FileNotFoundError('Legacy TRAINED.json and training_scaler.npz required alongside the experiment')
    return checkpoint,digest,metrics,trained,scaler


def evaluate_legacy_checkpoint(args):
    from exp.exp_pyramid_v2 import Exp_Pyramid_V2,signature_v2
    if args.is_training: raise ValueError('Legacy checkpoint route cannot train')
    if args.pyramid_skip_test: raise ValueError('Legacy evaluation requires test execution')
    checkpoint,digest,reference,trained,scaler=resolve_legacy(args)
    cfg,identity,_=signature_v2(args)
    folder=inside_project(args.pyramid_v2_output)/'compat_evaluations'/f'{identity}_{digest[:8]}'
    folder.mkdir(parents=True,exist_ok=True)
    origin=dict(evaluation_only=True,model=args.model,
                original_checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                original_checkpoint_sha256=digest,signature_config=cfg,
                note='Historical training metadata retained; no new training was performed.')
    with RunLock(folder/'run.lock'):
        origin_path=folder/'checkpoint_origin.json'
        if origin_path.exists():
            if json.loads(origin_path.read_text(encoding='utf-8'))!=origin:
                raise ValueError('Compatibility output provenance mismatch')
        else:
            if (folder/'checkpoints/best.pth').exists(): raise FileExistsError('Untracked compatibility checkpoint')
            atomic_json(origin,origin_path)
        (folder/'checkpoints').mkdir(exist_ok=True)
        for src,dst in [(checkpoint,folder/'checkpoints/best.pth'),(trained,folder/'TRAINED.json'),(scaler,folder/'training_scaler.npz')]:
            if dst.exists():
                if file_hash(src)!=file_hash(dst): raise ValueError('Compatibility copy differs from source')
            else: shutil.copyfile(src,dst)
        with open(folder/'evaluation.log','a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                try:
                    current=completed_metrics(folder)
                    if current is None:
                        local_args=copy.deepcopy(args)
                        local_args.pyramid_checkpoint=(folder/'checkpoints/best.pth').relative_to(ROOT).as_posix()
                        exp=Exp_Pyramid_V2(local_args)
                        # Validate training-only scaler before unchanged evaluate_best.
                        dataset,_=exp._get_data('test')
                        saved=np.load(scaler)
                        for k,value in [('mean',dataset.scaler.mean_),('scale',dataset.scaler.scale_),('var',dataset.scaler.var_)]:
                            np.testing.assert_array_equal(saved[k],value)
                        current=exp.evaluate_best(folder)
                    assert current['test_samples']==reference['test_samples']
                    for key in ('mse','mae'):
                        if not np.isclose(current[key],reference[key],atol=1e-7,rtol=1e-6):
                            raise ValueError(f'Reference metric mismatch: {key}: {current[key]} vs {reference[key]}')
                    if file_hash(checkpoint)!=digest: raise ValueError('Original checkpoint unexpectedly changed')
                    print(f'Legacy checkpoint evaluation PASS: model={args.model}, mse={current["mse"]}, mae={current["mae"]}')
                    print('Evaluation-only outputs: '+str(folder))
                    return current
                except BaseException:
                    record_failure(folder)
                    raise
