        for folder in good_folders:
            print(f"Inference at folder: {folder}")
            save_dict = torch.load(os.path.join(folder, "model.pkl"))
            train_args = NameSpace(save_dict["args"])
            random.seed(train_args.seed)
            np.random.seed(train_args.seed)
            torch.manual_seed(train_args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # load model
            algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
            algorithm = algorithm_class(
                dataset.input_shape,
                dataset.num_classes,
                len(dataset) - len(inf_args.test_envs),
                hparams=save_dict["model_hparams"]
            )
            algorithm._init_from_save_dict(save_dict)
            algorithm.to(device)

            ood_loaders = [
                FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
                for split in ood_names
            ]

            results = {}
            ood_evals = zip(ood_names, ood_loaders)
            for i, (name, loader) in enumerate(ood_evals):
                print(f"Inference at {name}")
                acc = algorithm.accuracy(
                    loader,
                    device,
                    compute_trace=False,
                    update_temperature=False,
                    output_temperature=(i == len(ood_names) - 1)
                )
                for key in acc:
                    results[name + f'_{key}'] = acc[key]

            results_keys = sorted(results.keys())
            printed_keys = [key for key in results_keys if "diversity" not in key.lower()]
            misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12, latex=True)
            misc.print_row([results[key] for key in printed_keys], colwidth=12, latex=True)
