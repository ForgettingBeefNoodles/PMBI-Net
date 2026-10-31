"""Local-only syntax, configuration, frozen checkpoint and forward checks; never train."""
import argparse,ast,hashlib,json,os,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from run import parse_args
from configs.pmbi_hparams import TASK_OVERRIDES,get_config
from models.PMBI_Net import Model
from exp.exp_pyramid import file_hash
from exp.exp_pyramid_v2 import Exp_Pyramid_V2,signature_v2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='checkpoints/reference_PEMS04_96/best.pth')
    parser.add_argument('--evaluate_reference',action='store_true')
    opt=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(4)
    report={'root':str(ROOT),'training_runs':0,'tasks':[]}
    evidence=json.loads((ROOT/'provenance/formal_configs.json').read_text())
    for folder in ['models','exp','utils','data_provider','configs','scripts']:
        for p in (ROOT/folder).glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    ast.parse((ROOT/'run.py').read_text(encoding='utf-8'))
    for dataset,h in TASK_OVERRIDES:
        a=parse_args(['--config','configs/pmbi_hparams.py','--exp_key',f'{dataset}_{h}'])
        original=evidence[str((dataset,h))]['original_args']
        ignored={'root_path','checkpoints','pyramid_output','pyramid_v2_output','pyramid_checkpoint','resume','is_training','model'}
        assert a.model=='PMBI_Net' and original['model']=='TQNet_Pyramid_V2'
        assert all(getattr(a,k)==v for k,v in original.items() if k not in ignored)
        for key in ['root_path','checkpoints','pyramid_output','pyramid_v2_output']:
            assert not Path(getattr(a,key)).is_absolute()
            assert (ROOT/getattr(a,key)).resolve().is_relative_to(ROOT)
        data=ROOT/a.root_path/a.data_path
        row={'dataset':dataset,'pred_len':h,'config_match':True,'data_present':data.is_file()}
        if data.is_file():
            row['data_sha_match']=file_hash(data)==evidence[str((dataset,h))]['data_sha256']
            assert row['data_sha_match']
        report['tasks'].append(row)
    a=parse_args(['--config','configs/pmbi_hparams.py','--exp_key','PEMS04_96','--use_gpu','false','--batch_size','2'])
    assert not a.use_gpu and a.batch_size==2
    a=parse_args(['--config','configs/pmbi_hparams.py','--exp_key','PEMS04_96']);a.use_gpu=False
    torch.manual_seed(2024);m=Model(a).float().eval()
    report['parameters']=sum(p.numel() for p in m.parameters());assert report['parameters']==1564320
    fixture=ROOT/'migration_checks/source_forward_fixture.npz'
    torch.manual_seed(734);x=torch.randn(2,96,307);phase=torch.tensor([0,287])
    with torch.inference_mode():y=m(x,phase)
    assert tuple(y.shape)==(2,96,307) and torch.isfinite(y).all()
    report['cpu_forward']='PASS'
    ck=(ROOT/opt.checkpoint).resolve()
    if not ck.is_relative_to(ROOT):raise ValueError('External checkpoint prohibited')
    if ck.is_file():
        weights=torch.load(ck,map_location='cpu',weights_only=True)
        m.load_state_dict(weights,strict=True)
        report['strict_checkpoint']='PASS';report['checkpoint_sha256']=file_hash(ck)
        assert list(weights)==list(m.state_dict())
        with torch.inference_mode():y=m(x,phase).numpy()
        if fixture.exists():
            f=np.load(fixture);np.testing.assert_array_equal(x.numpy(),f['x'])
            np.testing.assert_allclose(y,f['prediction'],atol=1e-6,rtol=0)
            report['source_forward_max_abs_diff']=float(np.max(np.abs(y-f['prediction'])))
        else:report['source_forward_comparison']='NOT VERIFIED: source fixture absent'
    else:report['strict_checkpoint']='NOT VERIFIED: checkpoint absent'
    if opt.evaluate_reference:
        if not ck.is_file():raise FileNotFoundError(ck)
        assert report['checkpoint_sha256']==evidence[str(('PEMS04',96))]['checkpoint_sha256']
        a.use_gpu=torch.cuda.is_available();exp=Exp_Pyramid_V2(a)
        exp.model.load_state_dict(weights,strict=True);exp.model.eval()
        dataset,loader=exp._get_data('test')
        assert not loader.drop_last and type(loader.sampler).__name__=='SequentialSampler'
        saved=np.load(ck.parent/'training_scaler.npz')
        for k,v in [('mean',dataset.scaler.mean_),('var',dataset.scaler.var_),('scale',dataset.scaler.scale_)]:np.testing.assert_array_equal(v,saved[k])
        squared=absolute=0.;count=windows=0
        with torch.inference_mode():
            for batch in loader:
                pred,true,_=exp.batch(batch)
                diff=pred.cpu().numpy().astype(np.float64)-true.cpu().numpy().astype(np.float64)
                squared+=(diff**2).sum();absolute+=np.abs(diff).sum();count+=diff.size;windows+=len(diff)
        metrics={'mse':float(squared/count),'mae':float(absolute/count),'test_samples':windows,'elements':count}
        old=evidence[str(('PEMS04',96))]['metrics']
        assert windows==old['test_samples']==3303
        assert abs(metrics['mse']-old['mse'])<1e-7 and abs(metrics['mae']-old['mae'])<1e-7
        assert file_hash(ck)==report['checkpoint_sha256']
        report['reference_evaluation']=metrics
    report['status']='PASS'
    folder=ROOT/'refactor_checks'/('portability_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));folder.mkdir(parents=True,exist_ok=False)
    (folder/'portability_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
