# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = [
        'Debug28', 'RotatedMNIST', 'ColoredMNIST', 'CustomColoredMNIST', 'CustomGrayColoredMNIST'
    ]
    MAX_EPOCH_5000 = dataset != 'CelebA_Blond'

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        # assert(name not in hparams)
        if name in hparams:
            print(
                f"Warning: parameter {name} was overriden from {hparams[name]} to {default_val, random_val_fn}."
            )
        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('model', None, lambda r: None)
    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('swa', 1, lambda r: r.choice([1]))
    _hparam('shared_init', 0, lambda r: r.choice([0]))

    if os.environ.get("HP") == "D":
        _hparam('resnet_dropout', 0., lambda r: 0.)
    elif os.environ.get("HP") == "W":
        _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5]))
    else:
        _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    _hparam('class_balanced', False, lambda r: False)
    _hparam('unfreeze_resnet_bn', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('beta1_d', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2**r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r: r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam(
            'irm_penalty_anneal_iters',
            500, lambda r: int(10**r.uniform(0, 4. if MAX_EPOCH_5000 else 3.5))
        )

    elif algorithm == "Mixup" or algorithm == "LISA":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))
        _hparam('mixup_proba', 1.0, lambda r: 1.)
        # 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_lambda', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam(
            'vrex_penalty_anneal_iters',
            500, lambda r: int(10**r.uniform(0, 4. if MAX_EPOCH_5000 else 3.5))
        )

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: int(10**r.uniform(-3, 5)))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    if algorithm in ['SAM']:
        _hparam('samadapt', 0, lambda r: r.choice([0]))
        _hparam('phosam', 0.05, lambda r: r.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]))

    if algorithm in ["Ensembling", "Ensemblingv2"]:
        _hparam("num_members", 2, lambda r: 2)
        _hparam('lr_ratio', 0, lambda r: r.choice([0]))
        _hparam('specialized', 0, lambda r: int(2**r.randint(0, 4)))

    # if algorithm == "Ensemblingv2":
    #     _hparam("lambda_ib_firstorder", 0, lambda r: 0)
    #     # for domain matching
    #     _hparam('lambda_domain_matcher', 0, lambda r: 10**r.uniform(1, 4))
    #     _hparam("similarity_loss", "none", lambda r: "none")
    #     # for fishr
    #     _hparam('ema', 0.95, lambda r: r.uniform(0.9, 0.99))
    #     _hparam('method', "weight", lambda r: r.choice([""]))

    # if algorithm in ["SWA"]:
    #     _hparam('swa', 1, lambda r: r.choice([1]))
    #     _hparam('split_swa', "", lambda r: r.choice([""]))
    #     _hparam('layerwise', "", lambda r: r.choice([""]))
    #     _hparam('swa_bin', 0.5, lambda r: r.uniform(0., 0.5))
    #     _hparam('swa_exp', 1, lambda r: r.choice([1]))

    # if algorithm in ["Subspace"]:
    #     _hparam('penalty_reg', 10**(-5), lambda r: r.choice([10**(-5), 10**(-6), 10**(-7)]))

    # if algorithm in ["Ensemblingv2", "SWA"]:
    #     if os.environ.get("HP") in ["D", "EoA"]:
    #         _hparam('penalty_anneal_iters', 1500, lambda r: 1500)
    #     else:
    #         _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000. if MAX_EPOCH_5000 else 2000)))
    #     _hparam("diversity_loss", "none", lambda r: "none")
    #     _hparam("div_data", "none", lambda r: "none")
    #     # for sampling diversity
    #     if os.environ.get("DIV") == "1":
    #         _hparam('div_eta', 0, lambda r: 10**r.uniform(-5, -2))
    #     elif os.environ.get("DIV") == "2":
    #         _hparam('div_eta', 0., lambda r: 10**r.uniform(-5, 0.))
    #     else:
    #         if os.environ.get("DIV") == "3":
    #             _hparam("lambda_diversity_loss", 0.0, lambda r: 10**r.uniform(-3, -1))
    #             _hparam("lambda_entropy", 0.0, lambda r: - 10**r.uniform(-3, -1))
    #         else:
    #             # for features diversity
    #             _hparam("conditional_d", False, lambda r: r.choice([False]))
    #             _hparam('clamping_value', 10, lambda r: r.choice([10]))
    #             _hparam('hidden_size', 64, lambda r: 64)  # 2**int(r.uniform(5., 7.)))
    #             _hparam('num_hidden_layers', 2., lambda r: r.choice([2]))
    #             _hparam('ib_space', "features", lambda r: r.choice(["features"]))
    #             _hparam('sampling_negative', "", lambda r: r.choice([""])) # "domain"

    #             _hparam("lambda_diversity_loss", 0.0, lambda r: 10**r.uniform(-3, -1))
    #             _hparam('weight_decay_d', 0.0005, lambda r: 0.0005)
    #             _hparam('reparameterization_var', 0.1, lambda r: 10**r.uniform(-3, 0))
    #             _hparam("lambda_entropy", 0.0, lambda r: 0)
    #             if dataset in SMALL_IMAGES:
    #                 _hparam('lr_d', 0.0005, lambda r: 10**r.uniform(-4.5, -2.5))
    #             else:
    #                 _hparam('lr_d', 0.0005, lambda r: 10**r.uniform(-4.5, -3.))
    # _hparam('lr_d', 0.0002, lambda r: 10**r.uniform(-4.5, -3.))
    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    # learning rate
    if os.environ.get("HP") in ["D", "EoA"]:
        _hparam('lr', 5e-5, lambda r: 5e-5)
    elif os.environ.get("HP") in ["S", "SE"]:
        _hparam('lr', 5e-5, lambda r: r.choice([1e-5, 3e-5, 5e-5]))
    elif os.environ.get("HP") in ["W"]:
        _hparam('lr', 5e-5, lambda r: r.choice([1e-5, 3e-5, 5e-5, 7e-5, 9e-5]))
    else:
        assert os.environ.get("HP", "Large") == "Large"
        if dataset == "Spirals":
            _hparam('lr', 0.01, lambda r: 10**r.uniform(-3.5, -1.5))
        elif dataset in SMALL_IMAGES:
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        elif algorithm == "LFF" and dataset == "ColoredMNISTLFF":
            # if algorithm in ['IRMAdv', "FisherMMD"]:
            #     _hparam('lr', 1e-3, lambda r: 10**r.uniform(-3.5, -2.))
            # else:
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        elif dataset == "ColoredMNISTLFF":
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        # elif dataset == "ColoredMNISTLFF":
        #     _hparam('lr', 1e-3, lambda r: 10**r.uniform(-5, -2))
        elif dataset == "BAR":
            _hparam("lr", 0.0001, lambda r: 0.0001)
        elif dataset == "Collage":
            _hparam("lr", 0.001, lambda r: 0.001)
        elif dataset == "TwoDirections2D":
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        else:
            _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    # if os.environ.get("LRD"):
    #     _hparam('lrdecay', 0.999, lambda r: 1. - 10**r.uniform(-5, -2))
    # else:
    _hparam('lrdecay', 0, lambda r: 0)

    if os.environ.get("HP") in ["D"]:
        _hparam('weight_decay', 0., lambda r: 0)
    elif os.environ.get("HP") in ["EoA", "SE"]:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -4))
    elif os.environ.get("HP") in ["S", "W"]:
        _hparam('weight_decay', 0., lambda r: r.choice([1e-4, 1e-6]))
    elif dataset == "Spirals":
        _hparam('weight_decay', 0.001, lambda r: 10**r.uniform(-6, -2))
    elif dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    # batch size
    if os.environ.get("HP") in ["1", "D", "EoA", "S", "SE"]:
        _hparam(
            'batch_size',
            int(os.environ.get("BS", 1)) * 32, lambda r: int(os.environ.get("BS", 1)) * 32
        )
    elif dataset == "Spirals":
        _hparam('batch_size', 512, lambda r: int(2**r.uniform(3, 9)))
    elif dataset == "ColoredMNISTLFF":
        _hparam('batch_size', 256, lambda r: 256)
    elif dataset == "BAR":
        _hparam('batch_size', 256, lambda r: 256)
    elif dataset == "Collage":
        _hparam('batch_size', 256, lambda r: 256)
    elif dataset in SMALL_IMAGES:
        _hparam(
            'batch_size',
            int(os.environ.get("BS", 1)) *
            64, lambda r: int(os.environ.get("BS", 1)) * int(2**r.uniform(3, 9))
        )
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    elif dataset == 'CelebA_Blond':
        _hparam('batch_size', 48, lambda r: int(2**r.uniform(4.5, 6)))
    elif dataset == "TwoDirections2D":
        _hparam('batch_size', 512, lambda r: 256)
    else:
        _hparam(
            'batch_size',
            int(os.environ.get("BS", 1)) *
            32, lambda r: int(os.environ.get("BS", 1)) * int(2**r.uniform(3, 5.5))
        )

    # if dataset == "Spirals":
    #     _hparam('mlp_width', 256, lambda r: int(2**r.uniform(6, 10)))
    #     _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))  # because linear classifier
    #     _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    if algorithm in ['DANN', 'CDANN']:
        if dataset in SMALL_IMAGES:
            _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
            _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
            _hparam('weight_decay_g', 0., lambda r: 0.)
        else:
            _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
            _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))
            _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    # model
    if dataset == "BAR":
        _hparam('model', "pretrained-resnet-18", lambda r: "pretrained-resnet-18")
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
