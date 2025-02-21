import datetime
import os
import time
import warnings
import copy 

import presets
import torch
import torch.utils.data
import torchvision
import utils
from coco_utils import get_coco
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode

import model_utils
import edgeai_torchmodelopt


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.SegmentationPresetTrain(base_size=args.base_size, crop_size=args.crop_size)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=args.base_size)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(args, model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    dataset_len = len(data_loader)
    with torch.inference_mode():
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]
            if args.val_epoch_size_factor and i >= round(args.val_epoch_size_factor * dataset_len):
                break

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat


def train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    dataset_len = len(data_loader)
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        if args.train_epoch_size_factor and i >= round(args.train_epoch_size_factor * dataset_len):
            break


def export_model(args, model, epoch, model_name):
    export_device = next(model.parameters()).device
    if args.quantization:
        if hasattr(model, "convert"):
            model = model.convert()
        else:
            model = torch.ao.quantization.quantize_fx.convert_fx(model)
        #
    #
    example_input = torch.rand((1,3,args.base_size,args.base_size), device=export_device)
    utils.export_on_master(model, example_input, os.path.join(args.output_dir, model_name), opset_version=args.opset_version)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # create logger that tee writes to file
    logger = edgeai_torchmodelopt.xnn.utils.TeeLogger(os.path.join(args.output_dir, 'run.log'))

    # weights can be an external url or a pretrained enum in torhvision
    (args.weights_url, args.weights_enum) = (args.weights, None) if edgeai_torchmodelopt.xnn.utils.is_url_or_file(args.weights) else (None, args.weights)
    
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(True, args))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )


    print("Creating model")
    model, surgery_kwargs = model_utils.get_model(args.model, weights=args.weights_enum, weights_backbone=args.weights_backbone, num_classes=num_classes, aux_loss=args.aux_loss, model_surgery=args.model_surgery)

    if args.model_surgery == edgeai_torchmodelopt.SyrgeryVersion.SURGERY_LEGACY:
        model = edgeai_torchmodelopt.xmodelopt.surgery.v1.convert_to_lite_model(model, **surgery_kwargs)
    elif args.model_surgery == edgeai_torchmodelopt.SyrgeryVersion.SURGERY_FX:
        model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model)

    if args.weights_url:
        print(f"loading pretrained checkpoint from: {args.weights_url}")
        edgeai_torchmodelopt.xnn.utils.load_weights(model, args.weights_url)

    if args.pruning == edgeai_torchmodelopt.PruningVersion.PRUNING_LEGACY:
        assert False, "Pruning is currently not supported in the legacy modules based method"
    elif args.pruning == edgeai_torchmodelopt.PruningVersion.PRUNING_FX:
        model = edgeai_torchmodelopt.xmodelopt.pruning.v2.PrunerModule(model)
    
    if args.quantization == edgeai_torchmodelopt.QuantizationVersion.QUANTIZATION_LEGACY:
        dummy_input = torch.rand(1,3,args.base_size,args.base_size)
        model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTrainModule(model, dummy_input=dummy_input, total_epochs=args.epochs)
    elif args.quantization == edgeai_torchmodelopt.QuantizationVersion.QUANTIZATION_FX:
        model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(model, total_epochs=args.epochs, qconfig_type=args.quantization_type)

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(args, model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(args, model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        export_model(args, model_without_ddp, epoch, model_name=args.model)

    logger.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    
    parser.add_argument(
        "--base-size", default=520, type=int, help="the size used for validation (default: 520)"
    )
    parser.add_argument(
        "--crop-size", default=480, type=int, help="the crop size used for training (default: 480)"
    )

    # options to create faster models
    parser.add_argument("--model-surgery", "--lite-model", default=0, type=int, choices=edgeai_torchmodelopt.SyrgeryVersion.get_choices(), help="model surgery to create lite models")

    parser.add_argument("--quantization", "--quantize", dest="quantization", default=0, type=int, choices=edgeai_torchmodelopt.QuantizationVersion.get_choices(), help="Quaantization Aware Training (QAT)")
    parser.add_argument("--quantization-type", default=None, help="Actual Quaantization Flavour - applies only if quantization is enabled")

    parser.add_argument("--pruning", default=0, help="Pruning/Sparsity")
    parser.add_argument("--pruning-ratio", default=0.5, help="Pruning/Sparsity Factor - applies only of pruning is enabled")
    parser.add_argument("--pruning-type", default=1, help="Pruning/Sparsity Type - applies only of pruning is enabled")

    parser.add_argument("--opset-version", default=18, help="ONNX Opset version")
    parser.add_argument("--train-epoch-size-factor", default=0.0, type=float,
                        help="Training validation breaks after one iteration - for quick experimentation")
    parser.add_argument("--val-epoch-size-factor", default=0.0, type=float,
                        help="Training validation breaks after one iteration - for quick experimentation")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
