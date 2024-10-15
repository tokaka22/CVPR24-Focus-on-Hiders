import argparse
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from models.preactresnet import PreActResNet18
from utils import *
from collections import OrderedDict
import shutil
import random
import wandb


upper_limit, lower_limit = 1,0
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./datasets/cifar10', type=str)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_one_drop', default=0.01, type=float)
    parser.add_argument('--lr_drop_epoch', default=100, type=int)
    parser.add_argument('--lr_proxy_max', default=0.01, type=float)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--attack_iters_test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd_alpha', default=2, type=float)
    parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--fgsm_alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm_init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='res/test00', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width_factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout_len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt_iters', default=10, type=int)
    parser.add_argument('--awp_gamma', default=0.01, type=float)
    parser.add_argument('--awp_warmup', default=0, type=int)

    # log
    parser.add_argument('--proj_name', type=str, default='Focus_test00', help='')
    parser.add_argument('--name', type=str, default='00', help='')
    parser.add_argument('--wd_offline', default=1, type=int)

    # auxiliary
    parser.add_argument('--aux_epsilon', default=7, type=int)
    parser.add_argument('--aux_attack_iters', default=7, type=int)
    parser.add_argument('--aux_pgd_alpha', default=7, type=float)
    parser.add_argument('--eps_gamma', type=float, default=1.0,
                    help='')
    parser.add_argument('--mean', type=float, default=0.4,
                    help='')
    parser.add_argument('--std_dev', type=float, default=0.015,
                    help='')
    
    parser.add_argument('--w_fix', default=0, type=int)
    parser.add_argument('--w_u', type=float, default=0.9,
                    help='')
    parser.add_argument('--w_r', type=float, default=0.1,
                    help='')
    parser.add_argument('--beta_u', type=float, default=2.0,
                    help='')
    parser.add_argument('--beta_r', type=float, default=0.5,
                    help='')
    parser.add_argument('--lt', type=float, default=1.1,
                    help='')                

    return parser.parse_args()


def main():
    args = get_args()

    args.eps_gamma = args.eps_gamma / 255.0

    print(args)

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project=args.proj_name, name=args.name, config=args)

    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    save_dir = os.path.join(args.fname, 'save')

    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        try:
            shutil.rmtree(save_dir)
            print(f"Directory '{save_dir}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{save_dir}' doesn't exist.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=8)
    else:
        dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=8)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=8)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    aux_epsilon = (args.aux_epsilon / 255.)
    aux_pgd_alpha = (args.aux_pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()

    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.lr_proxy_max)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]

    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(save_dir, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(save_dir, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(save_dir, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(save_dir, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(save_dir, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            # get auxiliary example
            if args.attack == 'pgd':
                # Random initialization
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta = attack_pgd(model, X, y, aux_epsilon, aux_pgd_alpha, args.aux_attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv_nonorm = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
            X_adv = normalize(X_adv_nonorm) # normalize TODO why?
            
            diff = X - X_adv_nonorm
            X_adv_eps_nonorm = X_adv_nonorm.detach()
            eps_beta = np.random.normal(args.mean, args.std_dev)
            wandb.log({'eps_beta': eps_beta})
            X_adv_eps = torch.clamp(X_adv_eps_nonorm + eps_beta * diff + args.eps_gamma * torch.randn_like(X).cuda(), min=lower_limit, max=upper_limit)
            X_adv_eps = normalize(X_adv_eps)
            
            model.train()

            # perturb it with the auxiliary example
            proxy.load_state_dict(model.state_dict())
            proxy.train()
            loss = nn.CrossEntropyLoss(reduction='none')(proxy(X_adv_eps), y)
            Indicator = (loss < args.lt).cuda().type(torch.cuda.FloatTensor)
            loss = -1 * (loss.mul(Indicator).mean())
            proxy_opt.zero_grad()
            loss.backward()
            proxy_opt.step()

            diff = diff_in_weights(model, proxy)
            add_into_weights(model, diff, coeff=1.0 * args.awp_gamma)

            u_robust_n_output = model(normalize(X))

            u_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_u_adv = normalize(torch.clamp(X + u_delta[:X.size(0)], min=lower_limit, max=upper_limit))

            u_robust_output = model(X_u_adv)
            loss_wv = criterion(u_robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            loss_wv.backward()

            add_into_weights(model, diff, coeff=-1.0 * args.awp_gamma)

            iter_cc_wv = 0
            grad_cc_wv = 0
            wv_gradient_dict = OrderedDict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wv_gradient_dict[name] = param.grad.clone()
                        grad_cc_wv = grad_cc_wv + 1
                    iter_cc_wv = iter_cc_wv + 1

            assert grad_cc_wv == iter_cc_wv

            robust_output = model(X_adv)
            loss_w = criterion(robust_output, y)

            opt.zero_grad()
            loss_w.backward()

            with torch.no_grad(): 
                robust_n_output = model(normalize(X))

                kl_robust = F.kl_div(F.log_softmax(robust_output, dim=1),
                                    F.softmax(robust_n_output, dim=1),
                                    reduction='sum')
                
                kl_u_robust = F.kl_div(F.log_softmax(u_robust_output, dim=1),
                                    F.softmax(u_robust_n_output, dim=1),
                                    reduction='sum')
                
                w_tensor = torch.stack([kl_robust, kl_u_robust])
                w_tensor[0] =  w_tensor[0] * args.beta_r
                w_tensor[1] =  w_tensor[1] * args.beta_u
                w_softmax = F.softmax(w_tensor, dim=0)

                w_r = w_softmax[0]
                w_u = w_softmax[1]

                wandb.log({'w_r': w_r, 'w_u': w_u})

            if args.w_fix:
                w_r = args.w_r
                w_u = args.w_u
                wandb.log({'w_r_fix': w_r, 'w_u_fix': w_u})

            iter_cc_r = 0
            grad_cc_r = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad = w_r * param.grad + w_u * wv_gradient_dict[name]
                        grad_cc_r = grad_cc_r + 1
                    iter_cc_r = iter_cc_r + 1

            assert grad_cc_r == iter_cc_r
                    
            opt.step()

            if args.mixup:
                loss = mixup_criterion(criterion, robust_n_output, y_a, y_b, lam)
            else:
                loss = criterion(robust_n_output, y)

            train_robust_loss += loss_w.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (robust_n_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

            wandb.log({'Epoch':epoch, 'LR': lr, 'Train Average Loss': train_loss/train_n, 'Train Average Acc': train_acc/train_n, 'Train Robust Average Loss': train_robust_loss/train_n, 'Train Robust Average Acc': train_robust_acc/train_n, 'Test Average Loss': test_loss/test_n, 'Test Average Acc': test_acc/test_n, 'Test Robust Average Loss': test_robust_loss/test_n, 'Test Robust Average Acc': test_robust_acc/test_n})

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(save_dir, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(save_dir, f'opt_{epoch}.pth'))

            # save best
            if (test_robust_acc/test_n > best_test_robust_acc) and (epoch > 80):
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(save_dir, f'model_best_{epoch}.pth'))
                best_test_robust_acc = test_robust_acc/test_n

                wandb.log({'best_test_robust_acc': test_robust_acc/test_n, 'best_test_acc': test_acc/test_n, 'epoch': epoch})
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
