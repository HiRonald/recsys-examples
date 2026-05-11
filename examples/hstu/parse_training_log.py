"""
解析 `training.py` 训练时打印到 stdout 的日志，抽取每个 `log_interval`（默认 100 步）
打印的 iter / loss，以及每次 eval 输出的 AUC，输出为一个 3 列（iter, loss, auc）
的纯文本文件，列与列之间用 Tab 分隔，可直接复制粘贴进 Excel。

只解析以下两类行（来自 `training/training.py`）：

  1) `[train] [iter 99, tokens 12345, elapsed_time 100.00 ms, achieved FLOPS 50.00 TFLOPS]: loss 0.123456`
     —— 每 `log_interval` 步打印一次（line 240-242）。
  2) `[eval] [eval 12800 users]:`
     `    Metrics.task0.AUC:0.654321`
     —— 每 `eval_interval` 步打印一次（line 75-78）。

每 10 步的简单 `epoch: ..., step: [...], loss = ...` 日志会被忽略，让 loss 的采样频率
与 AUC 一致。如确实希望保留每 10 步的细粒度 loss，可用 `--include-fine-loss`。

用法示例：

    # 1) 先在训练时把 stdout 落盘
    python -m training.training ... 2>&1 | tee train.log

    # 2.a) 默认：loss / auc 都按 log_interval（如 100）采样，每行同时有两个值
    python examples/hstu/parse_training_log.py train.log -o train_metrics.txt

    # 2.b) 细粒度 loss：保留每 10 步的 loss，AUC 仍每 100 步一次（其余行该列留空）
    python examples/hstu/parse_training_log.py train.log -o train_metrics_fine.txt \
        --include-fine-loss

输出 train_metrics.txt 的格式（默认）：

    iter    loss    auc
    99      0.5567  0.7234
    199     0.5123  0.7512
    ...

输出 train_metrics_fine.txt 的格式（--include-fine-loss）：

    iter    loss    auc
    0       0.7123
    10      0.6912
    ...
    99      0.5567  0.7234
    100     0.5550
    ...
    199     0.5123  0.7512

两者都用 Tab 分隔，直接全选复制到 Excel 即可被识别为 3 列。
"""

import argparse
import re
from collections import OrderedDict


# 形如:  epoch: 0, step: [10/3000], loss = 0.123456, time = 1.234s
SIMPLE_LOSS_RE = re.compile(
    r"epoch:\s*\d+,\s*step:\s*\[(\d+)/\d+\]\s*,\s*loss\s*=\s*([-+0-9.eE]+)"
)

# 形如:  [train] [iter 99, tokens 12345, elapsed_time 100.00 ms, achieved FLOPS 50.00 TFLOPS]: loss 0.123456
TRAIN_LOSS_RE = re.compile(
    r"\[train\]\s*\[iter\s+(\d+)[^\]]*\]\s*:\s*loss\s+([-+0-9.eE]+)"
)

# 形如:  Metrics.task0.AUC:0.654321  （也兼容大小写、可能没有 Metrics. 前缀）
AUC_RE = re.compile(r"AUC\s*[:=]\s*([-+0-9.eE]+)", re.IGNORECASE)


def parse_log(log_path: str, include_fine_loss: bool = False):
    """
    返回 OrderedDict[iter] = {"loss": float|None, "auc": float|None}，按 iter 升序。

    include_fine_loss=False (默认): 仅解析 `[train] [iter ...]: loss ...` 这种
        每 `log_interval` 步一次的日志，让 loss 与 AUC 的采样频率一致。
    include_fine_loss=True: 同时解析每 10 步的 `epoch: ..., step: ..., loss = ...`。
    """
    records: "OrderedDict[int, dict]" = OrderedDict()
    last_train_iter: int = None  # 用于把 eval 出来的 AUC 关联到最近一次训练 iter

    def get_rec(it: int) -> dict:
        rec = records.get(it)
        if rec is None:
            rec = {"loss": None, "auc": None}
            records[it] = rec
        return rec

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            # 匹配 print_rank_0 的格式化训练日志，每 `log_interval` 步一次。
            m = TRAIN_LOSS_RE.search(line)
            if m:
                it = int(m.group(1))
                loss = float(m.group(2))
                rec = get_rec(it)
                rec["loss"] = loss
                last_train_iter = it
                continue

            # 每 10 步的简单 print，默认跳过；仅在显式要求时才解析。
            if include_fine_loss:
                m = SIMPLE_LOSS_RE.search(line)
                if m:
                    it = int(m.group(1))
                    loss = float(m.group(2))
                    rec = get_rec(it)
                    if rec["loss"] is None:
                        rec["loss"] = loss
                    last_train_iter = it
                    continue

            # 匹配 eval 输出的 AUC，并挂到最近一次出现过的训练 iter 上。
            # 一次 eval 可能输出多个任务的 AUC，这里默认取最后一个解析到的值；
            # 如需保留所有任务，可自行扩展 records[it]["auc"] 的类型。
            m = AUC_RE.search(line)
            if m:
                auc = float(m.group(1))
                if last_train_iter is not None:
                    rec = get_rec(last_train_iter)
                    rec["auc"] = auc
                continue

    # 按 iter 升序
    return OrderedDict(sorted(records.items(), key=lambda kv: kv[0]))


def write_tsv(records, output_path: str, fill_blank: bool = True):
    """
    写入 3 列 Tab 分隔的文本文件。

    fill_blank=True 时，缺失值留空（Excel 会留空单元格）。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("iter\tloss\tauc\n")
        for it, rec in records.items():
            loss = "" if rec["loss"] is None else f"{rec['loss']:.6f}"
            auc = "" if rec["auc"] is None else f"{rec['auc']:.6f}"
            if not fill_blank and (loss == "" or auc == ""):
                continue
            f.write(f"{it}\t{loss}\t{auc}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parse HSTU training stdout log and emit a 3-column (iter/loss/auc) "
        "tab-separated text file you can paste into Excel."
    )
    parser.add_argument(
        "log",
        help="Path to the captured training stdout log file (e.g. saved via `tee train.log`).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="train_metrics.txt",
        help="Output text file path. Default: train_metrics.txt",
    )
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Only emit rows where both loss and auc are present (default: keep all rows, blanks allowed).",
    )
    parser.add_argument(
        "--include-fine-loss",
        action="store_true",
        help="Also parse the every-10-step `epoch: ..., step: ..., loss = ...` lines. "
        "By default only the `[train] [iter ...]: loss ...` lines (every log_interval) "
        "are used so loss aligns with the AUC sampling frequency.",
    )
    args = parser.parse_args()

    records = parse_log(args.log, include_fine_loss=args.include_fine_loss)
    write_tsv(records, args.output, fill_blank=not args.only_complete)

    n_total = len(records)
    n_loss = sum(1 for r in records.values() if r["loss"] is not None)
    n_auc = sum(1 for r in records.values() if r["auc"] is not None)
    print(
        f"Parsed {n_total} iters: {n_loss} with loss, {n_auc} with auc. "
        f"Wrote -> {args.output}"
    )


if __name__ == "__main__":
    main()
