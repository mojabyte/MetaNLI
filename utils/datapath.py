from os import path

loc = {
    "train": {
        "sc_de": "xnli/xnli.translate.train.clean.en-de.tsv",
        "sc_en": "xnli/train-en.tsv",
        "sc_es": "xnli/xnli.translate.train.clean.en-es.tsv",
        "sc_fa": "farstail/Train-word.csv",
        "sc_fr": "xnli/xnli.translate.train.clean.en-fr.tsv",
        "sc_en_0": "dreca/train-0.csv",
        "sc_en_1": "dreca/train-1.csv",
        "sc_en_2": "dreca/train-2.csv",
        "sc_en_3": "dreca/train-3.csv",
        "sc_en_4": "dreca/train-4.csv",
        "sc_en_5": "dreca/train-5.csv",
        "sc_en_6": "dreca/train-6.csv",
        "sc_en_7": "dreca/train-7.csv",
        "qa_en": "squad/train-v1.1.json",
        "qa_fa": "persianqa/pqa_train.json",
    },
    "dev": {
        "sc_de": "xnli/dev-de.tsv",
        "sc_en": "xnli/dev-en.tsv",
        "sc_es": "xnli/dev-es.tsv",
        "sc_fa": "farstail/Val-word.csv",
        "sc_fr": "xnli/dev-fr.tsv",
        "sc_en_0": "xnli/dev-en.tsv",
        "sc_en_1": "xnli/dev-en.tsv",
        "sc_en_2": "xnli/dev-en.tsv",
        "sc_en_3": "xnli/dev-en.tsv",
        "sc_en_4": "xnli/dev-en.tsv",
        "sc_en_5": "xnli/dev-en.tsv",
        "sc_en_6": "xnli/dev-en.tsv",
        "sc_en_7": "xnli/dev-en.tsv",
        "qa_en": "mlqa/MLQA_V1/dev/dev-context-en-question-en.json",
        "qa_fa": "persianqa/pqa_dev.json",
    },
    "test": {
        "sc_de": "xnli/test-de.tsv",
        "sc_en": "xnli/test-en.tsv",
        "sc_es": "xnli/test-es.tsv",
        "sc_fa": "farstail/Test-word.csv",
        "sc_fr": "xnli/test-fr.tsv",
        "qa_en": "mlqa/MLQA_V1/test/test-context-en-question-en.json",
        "qa_fa": "persianqa/pqa_test.json",
    },
}


def get_loc(type, task, base_dir="data/"):
    return path.join(base_dir, loc[type][task])
