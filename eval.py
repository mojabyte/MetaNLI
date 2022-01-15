import os, json, argparse, torch, logging, warnings, sys

from torch.utils.data import DataLoader

from model import MultiTaskModel
from utils.data import CorpusNLI, CorpusQA
from utils.datapath import get_loc
from utils.utils import evaluateNLI, evaluateQA
from utils.logger import Logger
from utils.seed import seed_everything


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")

parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)

parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")

parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")

parser.add_argument("--seed", type=int, default=63, help="seed for numpy and pytorch")
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="zeroshot_logs.txt", help="")
parser.add_argument("--grad_clip", type=float, default=1.0)

parser.add_argument("--task", type=str, default="sc_fa")

args = parser.parse_args()
print(args)

seed_everything(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(task_lang):
    [task,] = task_lang.split("_")

    if task == "sc":
        test_corpus = CorpusNLI(
            get_loc("test", task_lang, args.data_dir),
            model_name=args.model_name,
            local_files_only=args.local_model,
        )
        batch_size = args.sc_batch_size

    elif task == "qa":
        test_corpus = CorpusQA(
            get_loc("test", task_lang, args.data_dir),
            evaluate=True,
            model_name=args.model_name,
            local_files_only=args.local_model,
        )
        batch_size = args.qa_batch_size

    return test_corpus, batch_size


test_corpus, batch_size = load_data(args.task)
test_dataloader = DataLoader(
    test_corpus, batch_size=batch_size, pin_memory=True, drop_last=True
)

# Model
if args.load != "":
    print(f"loading model {args.load}...")
    model = torch.load(args.load)
else:
    model = MultiTaskModel(args).to(device)


def test():
    model.eval()

    if "sc" in args.task:
        test_loss, test_acc, matrix = evaluateNLI(
            model, test_dataloader, device, return_matrix=True
        )
        print("test_loss {:10.8f} test_acc {:6.4f}".format(test_loss, test_acc))
        print("confusion matrix:\n", matrix)

    elif "qa" in args.task:
        result = evaluateQA(model, test_corpus, "test_" + args.task, args.save)
        print("test_f1 {:10.8f}".format(result["f1"]))
        with open(os.path.join(args.save, "test.json"), "w") as outfile:
            json.dump(result, outfile)
        test_loss = -result["f1"]

    return test_loss


if __name__ == "__main__":
    test()
