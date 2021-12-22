import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
from sklearn.metrics import roc_auc_score


def main(iter):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=None)
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    # load checkpoint
    checkpoint_path = cfg.checkpoint.save_dir + f"checkpoint{iter}.pt"
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path) # or best?
    model.load_state_dict(
        state["model"], strict=True, model_cfg=cfg.model
    )
    del state["model"]

    model.to(torch.cuda.current_device())
    # load dataset
    split = "test"
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            output = model(**sample["net_input"])[:, 0, :]
            y = output.reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    auc = roc_auc_score(y_true, y_pred)
    print("auc = %f" % auc)

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    torch.save(y_pred, "y_pred.pt")
    torch.save(y_true, "y_true.pt")

    # evaluate
    evaluator = ogb.lsc.PCQM4MEvaluator()
    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    result_dict = evaluator.eval(input_dict)
    print('PCQM4MEvaluator:', result_dict)


if __name__ == '__main__':
    for iter in range(1, 9):
        main(iter)
