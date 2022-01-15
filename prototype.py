import argparse, time, torch, os, logging, warnings, sys

import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW

from models.sc_model import MultiTaskModel
from samplers.pt_sampler import TaskSampler
from learners.pt_learner import PtLearner
from losses import CPELoss
from utils.data import CorpusNLI
from utils.datapath import loc, get_loc
from utils.seed import seed_everything
from utils.logger import Logger


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--meta_lr", type=float, default=2e-5, help="Meta learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
parser.add_argument("--hidden_dims", type=int, default=768, help="")

parser.add_argument(
    "--lambda_1", type=float, default=1.0, help="DCE Coefficient in loss function"
)
parser.add_argument(
    "--lambda_2", type=float, default=0.5, help="CE Coefficient in loss function"
)
parser.add_argument(
    "--temp_scale",
    type=float,
    default=0.2,
    help="Temperature scale for DCE in loss function",
)

# bert-base-multilingual-cased
# xlm-roberta-base
parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)
parser.add_argument("--grad_clip", type=float, default=5.0)

parser.add_argument("--sc_labels", type=int, default=3, help="NLI labels count")
parser.add_argument("--sc_batch_size", type=int, default=32, help="NLI batch size")

# ---------------
parser.add_argument("--epochs", type=int, default=5, help="iterations")
parser.add_argument("--start_epoch", type=int, default=0, help="start iterations from")
parser.add_argument("--ways", type=int, default=3, help="number of ways")
parser.add_argument("--shot", type=int, default=4, help="number of shots")
parser.add_argument("--query_num", type=int, default=4, help="number of queries")
parser.add_argument(
    "--target_shot", type=int, default=0, help="number of target queries"
)
parser.add_argument("--meta_iteration", type=int, default=3000, help="")
# ---------------

parser.add_argument("--seed", type=int, default=42, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=200,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="main_output.txt", help="")
parser.add_argument("--meta_tasks", type=str, default="sc_en")
parser.add_argument("--target_task", type=str, default="")
parser.add_argument("--queue_len", default=8, type=int)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--pin_memory", action="store_true", help="")

# Optimizer
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
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

args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

print("target tasks: ", args.target_task)

task_types = args.meta_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

for tt in task_types:
    if "_" in tt:
        list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print("support tasks: ", list_of_tasks)


def evaluate(model, data, device):
    with torch.no_grad():
        total_loss = 0.0
        for batch in data:
            output = model.forward(batch)
            data_labels = batch["label"].to(device)
            loss = F.cross_entropy(output[2], data_labels, reduction="none")
            loss = loss.detach().mean().item()
            total_loss += loss
        total_loss /= len(data)
        return total_loss


def evaluateMeta(model, dev_loaders, device):
    loss_dict = {}
    total_loss = 0
    tasks = [args.target_task] if args.target_task != "" else list_of_tasks
    model.eval()
    for i, task in enumerate(tasks):
        loss = evaluate(model, dev_loaders[i], device)
        loss_dict[task] = loss
        total_loss += loss
    return loss_dict, total_loss


def main():
    seed_everything(args.seed)

    # Prepare train and validation dataloaders
    train_loaders = []
    dev_loaders = []

    for task in list_of_tasks:
        train_corpus = CorpusNLI(
            get_loc("train", task, args.data_dir),
            model_name=args.model_name,
            local_files_only=args.local_model,
        )
        train_sampler = TaskSampler(
            train_corpus,
            n_way=args.ways,
            n_shot=args.shot,
            n_query=args.query_num,
            n_tasks=args.meta_iteration,
        )
        train_loader = DataLoader(
            train_corpus,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=train_sampler.episodic_collate_fn,
            shuffle=False,
        )
        train_loaders.append(train_loader)

        if args.target_task == "":
            dev_corpus = CorpusNLI(
                get_loc("dev", task, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_loader = DataLoader(
                dev_corpus, batch_size=args.sc_batch_size, pin_memory=args.pin_memory
            )
            dev_loaders.append(dev_loader)

    if args.target_task != "":
        ### == target dataset ==============
        trg_train_corpus = CorpusNLI(
            get_loc("train", args.target_task, args.data_dir),
            model_name=args.model_name,
            local_files_only=args.local_model,
        )
        trg_dev_corpus = CorpusNLI(
            get_loc("dev", args.target_task, args.data_dir),
            model_name=args.model_name,
            local_files_only=args.local_model,
        )

        trg_train_sampler = TaskSampler(
            trg_train_corpus,
            n_way=args.ways,
            n_shot=0,
            n_query=args.target_shot,
            n_tasks=args.meta_iteration,
        )
        trg_train_loader = DataLoader(
            trg_train_corpus,
            batch_sampler=trg_train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=trg_train_sampler.episodic_collate_fn,
        )
        trg_dev_loader = DataLoader(
            trg_dev_corpus, batch_size=args.sc_batch_size, pin_memory=args.pin_memory
        )
        dev_loaders = [trg_dev_loader]

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
    criterion = CPELoss(device, args)

    ## == training ======
    log_interval_time = time.time()

    min_task_losses = {
        "sc": float("inf"),
        "qa": float("inf"),
    }

    pt_learner = PtLearner(criterion, device)

    for epoch in range(args.start_epoch, args.epochs):
        print(f"======================= Epoch {epoch} =======================")
        train_loss = 0.0

        train_loader_iterations = [iter(train_loader) for train_loader in train_loaders]

        if args.target_task != "":
            trg_train_loader_iteration = iter(trg_train_loader)

        for iteration in range(args.meta_iteration):

            # == Data preparation ===========
            queue = [next(trainloader) for trainloader in train_loader_iterations]

            if args.queue_len < len(train_loader_iterations):
                queue = random.sample(queue, args.queue_len)

            trg_queue = []
            if args.target_task != "":
                trg_queue = [next(trg_train_loader_iteration)]

            ## == train ===================
            loss = pt_learner.train(model, queue, trg_queue, optim, iteration, args)
            train_loss += loss

            ## == validation ==============
            if (iteration + 1) % args.log_interval == 0:
                total_loss = train_loss / args.log_interval
                train_loss = 0.0

                # evalute on val_dataset
                val_loss_dict, val_loss_total = evaluateMeta(
                    model, dev_loaders, device=device
                )

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

                total_loss = 0

            if args.scheduler:
                scheduler.step()

    print("Saving new last model...")
    torch.save(model, os.path.join(args.save, "model_last.pt"))


if __name__ == "__main__":
    main()
