import argparse
import torch
import numpy as np

from utils import get_datasets, imagenet_calibration_datasets, load_checkpoint
from utils import one_shot_prune, one_shot_uniform_prune, one_shot_NM_prune, test_sparsity
from utils.eval_utils import eval_model, fix_bn_stats
from torch.utils.data import DataLoader


def get_parser():
    parser = argparse.ArgumentParser(description='One shot pruning of SAM trained models')
    parser.add_argument('--cpu', action='store_true', help='force model on CPU')
    parser.add_argument('--unif', action='store_true', help='use uniform pruning; uses global if false')
    parser.add_argument('--use_calib', action='store_true',
                        help='use calibration dataset for BN tuning; uses the entire train set if false')
    parser.add_argument('--fixed_calib', action='store_true', help='use FIXED calibration dataset for BN tuning')
    parser.add_argument('--calib_size', type=int, default=1000, help='# samples or # samples / class (fixed calibration) for the calibration set')
    parser.add_argument('--seed', type=int, default=42, help='seed for calibration loader and/or subset')
    parser.add_argument('--pruning', type=str, default='global', help='type of pruning to use (global | uniform | NM)')
    parser.add_argument('--N', type=int, default=2, help='for N:M pruning')
    parser.add_argument('--M', type=int, default=4, help='for N:M pruning')
    parser.add_argument('--sparsity', type=float, default=0.8, help='target sparsity (between 0. and 1.)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        DEV = torch.device('cuda:0')
        torch.cuda.manual_seed(args.seed)
    else:
        DEV = torch.device('cpu')

    # args.use_calib should be True (otherwise it uses the entire training set for BNT calibration
    if args.use_calib:
        if args.fixed_calib:
            # use fixed calibration set for BNT (only for ImageNet)
            # first call generate_calibration_dset.py to create the calibration set
            # args.calib_size should be 1 here
            data_train, data_test = imagenet_calibration_datasets('PATH/TO/CALIB', 'IMAGENET_PATH',
                                                                  calib_size=args.calib_size)
        else:
            # get a random calibration set (e.g. 1000 samples)
            data_train, data_test = get_datasets('imagenet', 'IMAGENET_PATH', use_val_split=False)
            idxs = np.arange(len(data_train))
            np.random.shuffle(idxs)
            data_train = torch.utils.data.Subset(data_train, idxs[:args.calib_size])
    else:
        data_train, data_test = get_datasets('imagenet', 'IMAGENET_PATH', use_val_split=False)
    print('train', data_train)
    print('test', data_test)
    train_loader = DataLoader(data_train, batch_size=128, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, shuffle=False, pin_memory=True, num_workers=8)

    # load the dense model
    model_path = '/PATH/TO/CHECKPOINT'
    model = load_checkpoint(model_path, only_model=True)
    model.to(DEV)

    print('Evaluating the dense model...')
    eval_loss_dense, eval_acc_dense = eval_model(model, test_loader, DEV)
    print(f'ACC:{100*eval_acc_dense}    Loss: {eval_loss_dense}')

    if args.pruning=='global':
        print(f'Pruning the model globally to {100*args.sparsity}% sparsity')
        masks = one_shot_prune(model, args.sparsity)
    elif args.pruning=='uniform':
        print(f'Pruning the model uniformly to {100*args.sparsity}% sparsity')
        masks = one_shot_uniform_prune(model, args.sparsity)
    elif args.pruning=='NM':
        print(f'Pruning the model using {args.N}:{args.M} pattern to {100*args.N/args.M}% sparsity')
        masks = one_shot_NM_prune(model, args.N, args.M)
    else:
        raise NotImplementedError('Pruning distribution should be global | uniform | NM')
    sparsity = test_sparsity(model)
    print(f'Sparsity after pruning: {sparsity}')
        
    print('Evaluating the model after one-shot pruning...')
    eval_loss_sp, eval_acc_sp = eval_model(model, test_loader, DEV)
    print(f'ACC:{100*eval_acc_sp}    Loss: {eval_loss_sp}')

    print('Applying batch norm tuning to the sparse model...')
    fix_bn_stats(model, train_loader, DEV, iters=100)
    print(f'Evaluating model after BNT')
    eval_loss_BNT, eval_acc_BNT = eval_model(model, test_loader, DEV)
    print(f'ACC-BNT: {100*eval_acc_BNT}    Loss BNT: {eval_loss_BNT}')
