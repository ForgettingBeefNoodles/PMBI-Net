"""PMBI-Net portable entry. CLI > task configuration > original defaults."""
import argparse
import importlib.util
import os
from pathlib import Path
import random
import numpy as np
import torch
PROJECT_ROOT = Path(__file__).resolve().parent

def parse_bool(value):
    if isinstance(value,bool): return value
    if value.lower() in ('true','1','yes'): return True
    if value.lower() in ('false','0','no'): return False
    raise argparse.ArgumentTypeError('Expected true/false or 1/0')

def build_parser():
    parser = argparse.ArgumentParser(description='PMBI-Net formal model')
    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='TQNet',
                        help='PMBI_Net (formal) or TQNet_Pyramid_V2 (compatibility reference)')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # TQNet & CycleNet
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # SegRNN
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=parse_bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    from exp.exp_pyramid import add_arguments
    add_arguments(parser)
    from exp.exp_pyramid_v2 import add_arguments as add_v2_arguments
    add_v2_arguments(parser)
    parser.add_argument('--config', default='')
    parser.add_argument('--exp_key', default='')
    parser.add_argument('--check_config', action='store_true', help='Print validated args without training')
    return parser

def parse_args(argv=None):
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config',default='');probe.add_argument('--exp_key',default='')
    selected,_=probe.parse_known_args(argv)
    parser=build_parser()
    if selected.config:
        path=(PROJECT_ROOT/selected.config).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file(): parser.error('Config must be an existing project-local file')
        if not selected.exp_key: parser.error('--config requires --exp_key')
        spec=importlib.util.spec_from_file_location('pmbi_selected_config',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        try:
            dataset,horizon=selected.exp_key.rsplit('_',1)
            defaults=module.get_config(dataset,int(horizon))
        except (KeyError,ValueError) as exc: parser.error(str(exc))
        actions={action.dest:action for action in parser._actions}
        unknown=set(defaults)-set(actions)
        if unknown: parser.error('Unknown config parameters: '+str(sorted(unknown)))
        for key,value in defaults.items():
            action=actions[key]
            if action.choices is not None and value not in action.choices: parser.error('Invalid configured '+key)
            if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)) and not isinstance(value,bool): parser.error('Boolean required: '+key)
            if action.type is not None: defaults[key]=action.type(value)
            action.required=False
        parser.set_defaults(**defaults)
    elif selected.exp_key: parser.error('--exp_key requires --config')
    args=parser.parse_args(argv)
    for name in ('root_path','checkpoints','pyramid_output','pyramid_v2_output','pyramid_checkpoint','config'):
        value=getattr(args,name,'')
        if value:
            path=(PROJECT_ROOT/value).resolve()
            if not path.is_relative_to(PROJECT_ROOT): parser.error('Cross-project path prohibited: '+name)
            setattr(args,name,path.relative_to(PROJECT_ROOT).as_posix())
    if args.model not in ('PMBI_Net','TQNet_Pyramid_V2') or getattr(args,'pyramid_v2_variant','v2_full')!='v2_full': parser.error('This distribution exposes only formal PMBI-Net v2_full')
    args.pyramid_v2_experiment=True
    return args

def main(argv=None):
    os.chdir(PROJECT_ROOT)
    args=parse_args(argv)
    if args.check_config:
        import json
        print(json.dumps(vars(args),indent=2));return
    random.seed(args.random_seed);torch.manual_seed(args.random_seed);np.random.seed(args.random_seed)
    args.use_gpu=bool(torch.cuda.is_available() and args.use_gpu)
    print('Args in experiment:');print(args)
    from exp.exp_pyramid_v2 import run_experiment_v2
    run_experiment_v2(args)

if __name__=='__main__': main()
