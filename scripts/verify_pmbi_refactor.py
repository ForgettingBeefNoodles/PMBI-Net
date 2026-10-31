"""Old V2 vs PMBI-Net: real checkpoint, identical RNG, paired full-test regression.
No optimizer, weight updates, training, or writes to original artifacts.
"""
import argparse,ast,copy,datetime,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from run import parse_args
from models.TQNet_Pyramid_V2 import Model as Old
from models.PMBI_Net import Model as New
from exp.exp_pyramid import file_hash
from exp.exp_pyramid_v2 import Exp_Pyramid_V2
from exp.checkpoint_compat import resolve_legacy

def seed():
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2024)

def diff(x,y):
    return dict(max_abs_diff=float((x-y).abs().max()),mean_abs_diff=float((x-y).abs().mean()),
                allclose=bool(torch.allclose(x,y,rtol=0,atol=1e-6)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/reference_PEMS04_96/best.pth')
    p.add_argument('--skip_full_test',action='store_true');p.add_argument('--out',default='')
    opt=p.parse_args();os.chdir(ROOT);torch.set_num_threads(4)
    out=(ROOT/(opt.out or 'refactor_checks/test_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Need a fresh project-local report directory')
    out.mkdir(parents=True)
    args=parse_args(['--config','configs/pmbi_hparams.py','--exp_key','PEMS04_96','--is_training','0'])
    report={'training_runs':0,'optimizer_steps':0,'A':{},'B':'NOT TESTED','C':'NOT TESTED','D':'NOT TESTED'}
    # Verify the actual Model class and helper AST, not just their output shapes.
    def node(path,kind,name):return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,kind) and n.name==name)
    assert ast.dump(node('models/TQNet_Pyramid_V2.py',ast.ClassDef,'Model'))==ast.dump(node('models/PMBI_Net.py',ast.ClassDef,'Model'))
    for fn in ('preserve_rng','mlp'):
        assert ast.dump(node('models/TQNet_Pyramid.py',ast.FunctionDef,fn))==ast.dump(node('utils/pmbi_helpers.py',ast.FunctionDef,fn))
    seed();old=Old(args);rng_old=torch.get_rng_state().clone()
    seed();new=New(args);rng_new=torch.get_rng_state().clone()
    assert torch.equal(rng_old,rng_new)
    so,sn=old.state_dict(),new.state_dict();assert list(so)==list(sn)
    for k in so:
        assert so[k].shape==sn[k].shape and so[k].dtype==sn[k].dtype and torch.equal(so[k],sn[k]),k
    assert {k:v.requires_grad for k,v in old.named_parameters()}=={k:v.requires_grad for k,v in new.named_parameters()}
    counts=[sum(p.numel() for p in m.parameters()) for m in (old,new)];assert counts==[1564320,1564320]
    report['A']=dict(status='PASSED',parameters_old=counts[0],parameters_new=counts[1],state_dict_keys=len(so),
                     initialization_exact=True,rng_after_initialization_equal=True,model_and_helper_AST_identical=True)
    checkpoint=(ROOT/opt.checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT):raise ValueError('Checkpoint must be project-local')
    if checkpoint.is_file():
        args.pyramid_checkpoint=checkpoint.relative_to(ROOT).as_posix()
        checkpoint,digest,reference,_,scaler_path=resolve_legacy(args)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        ro=old.load_state_dict(weights,strict=True);rn=new.load_state_dict(weights,strict=True)
        report['B']=dict(status='PASSED',checkpoint=checkpoint.relative_to(ROOT).as_posix(),sha256=digest,
                         old_missing=ro.missing_keys,old_unexpected=ro.unexpected_keys,new_missing=rn.missing_keys,new_unexpected=rn.unexpected_keys)
        torch.manual_seed(73);x=torch.randn(2,96,307);phase=torch.tensor([0,287])
        old.eval();new.eval()
        with torch.inference_mode():result=diff(old(x,phase),new(x,phase))
        assert result['allclose'];report['C']=dict(status='PASSED',device='cpu',**result)
        # Dropout behavior and gradients under identical RNG, without optimization.
        old.train();new.train();old.zero_grad();new.zero_grad()
        seed();yo=old(x,phase);yo.square().mean().backward()
        seed();yn=new(x,phase);yn.square().mean().backward()
        assert torch.equal(yo,yn)
        max_grad=0.
        for (k,po),(l,pn) in zip(old.named_parameters(),new.named_parameters()):
            assert k==l and (po.grad is None)==(pn.grad is None)
            if po.grad is not None:
                assert torch.isfinite(po.grad).all() and torch.isfinite(pn.grad).all()
                max_grad=max(max_grad,float((po.grad-pn.grad).abs().max()))
        assert max_grad==0
        report['dropout_backward']=dict(status='PASSED',prediction_max_abs_diff=0.,gradient_max_abs_diff=max_grad,optimizer_steps=0)
        old.eval();new.eval()
        if not opt.skip_full_test and (ROOT/args.root_path/args.data_path).is_file():
            args.use_gpu=torch.cuda.is_available();old_args=copy.deepcopy(args);old_args.model='TQNet_Pyramid_V2'
            exp_old=Exp_Pyramid_V2(old_args);exp_new=Exp_Pyramid_V2(args)
            assert isinstance(exp_old.model,Old) and isinstance(exp_new.model,New)
            for exp in (exp_old,exp_new):exp.model.load_state_dict(weights,strict=True);exp.model.eval()
            data,loader=exp_new._get_data('test');data_old,loader_old=exp_old._get_data('test')
            assert not loader.drop_last and type(loader.sampler).__name__=='SequentialSampler'
            np.testing.assert_array_equal(data.data_x,data_old.data_x)
            saved=np.load(scaler_path)
            for k,v in [('mean',data.scaler.mean_),('scale',data.scaler.scale_),('var',data.scaler.var_)]:np.testing.assert_array_equal(v,saved[k])
            totals=np.zeros((2,2),dtype=np.float64);count=windows=0;max_diff=0.
            with torch.inference_mode():
                for batch,other in zip(loader,loader_old):
                    for j in (0,1,4):torch.testing.assert_close(batch[j],other[j],atol=0,rtol=0)
                    po,truth,_=exp_old.batch(other);pn,tn,_=exp_new.batch(batch)
                    assert torch.equal(truth,tn)
                    max_diff=max(max_diff,float((po-pn).abs().max()))
                    for i,pred in enumerate((po,pn)):
                        e=pred.cpu().numpy().astype(np.float64)-truth.cpu().numpy().astype(np.float64)
                        totals[i,0]+=(e*e).sum();totals[i,1]+=np.abs(e).sum()
                    count+=truth.numel();windows+=len(truth)
            totals/=count
            assert windows==len(data)==reference['test_samples']==3303 and max_diff<=1e-6
            np.testing.assert_allclose(totals[0],totals[1],atol=0,rtol=0)
            np.testing.assert_allclose(totals[1],[reference['mse'],reference['mae']],atol=1e-7,rtol=0)
            report['D']=dict(status='PASSED',test_samples=windows,elements=count,device=str(exp_new.device),
                             old_mse=totals[0,0],old_mae=totals[0,1],new_mse=totals[1,0],new_mae=totals[1,1],max_prediction_abs_diff=max_diff)
        assert file_hash(checkpoint)==digest
    else: print('CHECKPOINT TEST NOT RUN')
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('Report:',out)
if __name__=='__main__':main()
