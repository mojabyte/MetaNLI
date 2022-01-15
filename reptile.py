import argparse, time, torch, os, logging, warnings, sys

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW

from model import MultiTaskModel
from samplers.reptile_sampler import TaskSampler
from samplers.batch_sampler import UniformBatchSampler
from learners.reptile_learner import reptile_learner
from utils.data import CorpusNLI, CorpusQA
from utils.datapath import loc, get_loc
from utils.seed import seed_everything
from utils.logger import Logger


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--meta_lr", type=float, default=2e-5, help="Meta learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
parser.add_argument("--hidden_dims", type=int, default=768, help="")

# bert-base-multilingual-cased
# xlm-roberta-base
parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="Name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="Use local pretrained model"
)
parser.add_argument("--grad_clip", type=float, default=5.0)

parser.add_argument("--sc_labels", type=int, default=3, help="NLI labels count")
parser.add_argument("--qa_labels", type=int, default=2, help="QA labels count")

parser.add_argument("--sc_batch_size", type=int, default=32, help="NLI batch size")
parser.add_argument("--qa_batch_size", type=int, default=8, help="QA batch size")

parser.add_argument(
    "--update_step", type=int, default=3, help="number of Reptile update steps"
)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=1.0, help="")

# ---------------
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs from")
parser.add_argument("--ways", type=int, default=2, help="number of ways")
parser.add_argument("--shot", type=int, default=4, help="number of shots")
parser.add_argument("--meta_iterations", type=int, default=3000, help="")
# ---------------

parser.add_argument(
    "--val_interval",
    type=int,
    default=200,
    help="Validate after every val_interval iterations",
)
parser.add_argument("--meta_tasks", type=str, default="sc_en")
parser.add_argument("--queue_len", default=8, type=int)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--pin_memory", action="store_true", help="")

# Optimizer
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay for Adam optimizer"
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer"
)

# Scheduler options
parser.add_argument("--scheduler", action="store_true", help="Use scheduler")
parser.add_argument(
    "--step_size", default=3000, type=int, help="Step size for scheduler"
)
parser.add_argument(
    "--last_step", default=0, type=int, help="Last step of the scheduler"
)
parser.add_argument(
    "--gamma", default=0.1, type=float, help="Multiplicative factor of the scheduler"
)

parser.add_argument("--seed", type=int, default=63, help="seed for numpy and pytorch")
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="train_logs.txt", help="")


args = parser.parse_args()
print(args)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))

task_types = args.meta_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

for tt in task_types:
    if "_" in tt:
        list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print(list_of_tasks)


def evaluate(model, task, data):
    with torch.no_grad():
        total_loss = 0.0
        for batch in data:
            output = model.forward(task, batch)
            loss = output[0].detach().mean()
            total_loss += loss.item()
        total_loss /= len(data)
        return total_loss


def evaluateMeta(model, dev_loaders):
    loss_dict = {}
    total_loss = 0
    model.eval()
    for task in list_of_tasks:
        loss = evaluate(model, task, dev_loaders[task])
        loss_dict[task] = loss
        total_loss += loss
    return loss_dict, total_loss


def main():
    seed_everything(args.seed)

    # Prepare train and validation dataloaders
    train_loaders = {}
    dev_loaders = {}
    corpus_len = {}

    for task in list_of_tasks:
        time_dataloader = time.time()
        print(f"preparing {task} dataloaders...")

        train_corpus = None
        dev_corpus = None
        batch_size = 32

        if "sc" in task:
            train_corpus = CorpusNLI(
                get_loc("train", task, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusNLI(
                get_loc("dev", task, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.sc_batch_size

        elif "qa" in task:
            train_corpus = CorpusQA(
                get_loc("train", task, args.data_dir),
                evaluate=False,
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusQA(
                get_loc("dev", task, args.data_dir),
                evaluate=True,
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.qa_batch_size

        else:
            continue

        train_sampler = TaskSampler(
            train_corpus,
            n_way=args.ways,
            n_shot=args.shot,
            n_tasks=args.meta_iterations,
            reptile_step=args.update_step,
        )
        train_loader = DataLoader(
            train_corpus,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        train_loaders[task] = train_loader
        corpus_len[task] = len(train_corpus)

        dev_loader = DataLoader(
            dev_corpus, batch_size=batch_size, pin_memory=args.pin_memory
        )
        dev_loaders[task] = dev_loader

        print(f"Completed in {time.time() - time_dataloader:.2f}s.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    if args.load != "":
        print(f"loading model {args.load}...")
        model = torch.load(args.load)
    else:
        model = MultiTaskModel(args).to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.meta_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.meta_lr,
        },
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)

    # Scheduler
    scheduler = StepLR(
        optim,
        step_size=args.step_size,
        gamma=args.gamma,
        last_epoch=args.last_step - 1,
    )

    min_task_losses = {
        "sc": float("inf"),
        "qa": float("inf"),
    }

    sampler = UniformBatchSampler(
        train_loaders,
        corpus_len,
        list_of_tasks,
        temp=args.temp,
        queue_len=args.queue_len,
    )

    for epoch in range(args.start_epoch, args.epochs):
        print(f"======================= Epoch {epoch} =======================")
        train_loss = 0.0

        log_interval_time = time.time()

        for iteration, queue in enumerate(sampler):
            if iteration >= args.meta_iterations:
                break

            ## == Train ===================
            loss = reptile_learner(model, queue, optim, iteration, args)
            train_loss += loss

            ## == Validation ==============
            if (iteration + 1) % args.val_interval == 0:
                total_loss = train_loss / args.val_interval
                train_loss = 0.0

                # Evalute on the validation dataset
                val_loss_dict, val_loss_total = evaluateMeta(model, dev_loaders)

                loss_per_task = {}
                for task in val_loss_dict.keys():
                    if task[:2] in loss_per_task.keys():
                        loss_per_task[task[:2]] = (
                            loss_per_task[task[:2]] + val_loss_dict[task]
                        )
                    else:
                        loss_per_task[task[:2]] = val_loss_dict[task]

                for task in loss_per_task.keys():
                    if loss_per_task[task] < min_task_losses[task]:
                        print("Saving " + task + " Model")
                        torch.save(
                            model, os.path.join(args.save, "model_" + task + ".pt"),
                        )
                        min_task_losses[task] = loss_per_task[task]

                print(
                    f"Time: {time.time() - log_interval_time:.4f}, Step: {iteration + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss_total:.4f}"
                )
                log_interval_time = time.time()

            if args.scheduler:
                scheduler.step()

    print("Saving new last model...")
    torch.save(model, os.path.join(args.save, "model_last.pt"))


if __name__ == "__main__":
    main()
