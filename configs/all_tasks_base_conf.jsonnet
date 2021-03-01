local bert_model_name = "bert-base-cased";
local embedding_dim = 768;
local dropout = 0.1;
local batch_size = 8;
local output_dir = "results/all_tasks_base_conf/";
local server = true;
local data_path = if server then "/srv/scratch0/jgoldz/mthesis/data/" else "/home/janis/Dropbox/UZH/UFSP_Digital_Religion/Master_Thesis/thesis_code/data/";
local cuda_device = 6;

local reader_common = {
        "max_sequence_length": 512,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model_name
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model_name
            }
        }
};

{
    "dataset_reader": {
        "type": "multitask",
        "readers": {
            "SemEval2016": reader_common {
                "type": "SemEval2016"
            },
            "IBMCS": reader_common {
                "type": "IBMCS"
            },
            "arc": reader_common {
                "type": "arc"
            },
            "ArgMin": reader_common {
                "type": "ArgMin"
            },
            "FNC1": reader_common {
                "type": "FNC1"
            },
            "IAC": reader_common {
                "type": "IAC"
            },
            "PERSPECTRUM": reader_common {
                "type": "PERSPECTRUM"
            },
            "SCD": reader_common {
                "type": "SCD"
            },
            "SemEval2019": reader_common {
                "type": "SemEval2019"
            },
            "Snopes": reader_common {
                "type": "Snopes"
            }
        }
    },
    "data_loader": {
        "type": "multitask",
        "scheduler": {
            "type": "homogeneous_roundrobin",
            "batch_size": batch_size
        }
    },
    "train_data_path": {
        "SemEval2016": data_path + "en/SemEval2016Task6/train.jsonl",
        "IBMCS": data_path + "en/IBM_CLAIM_STANCE/train.jsonl",
        "arc": data_path + "en/arc/train.jsonl",
        "ArgMin": data_path + "en/ArgMin/train.jsonl",
        "FNC1": data_path + "en/fnc-1/train.jsonl",
        "IAC": data_path + "en/IAC/train.jsonl",
        "PERSPECTRUM": data_path + "en/PERSPECTRUM/train.jsonl",
        "SCD": data_path + "en/SCD/train.jsonl",
        "SemEval2019": data_path + "en/SemEval2019Task7/train.jsonl",
        "Snopes": data_path + "en/Snopes/train.jsonl"
    },
    "validation_data_path": {
        "SemEval2016": data_path + "en/SemEval2016Task6/dev.jsonl",
        "IBMCS": data_path + "en/IBM_CLAIM_STANCE/dev.jsonl",
        "arc": data_path + "en/arc/dev.jsonl",
        "ArgMin": data_path + "en/ArgMin/dev.jsonl",
        "FNC1": data_path + "en/fnc-1/dev.jsonl",
        "IAC": data_path + "en/IAC/dev.jsonl",
        "PERSPECTRUM": data_path + "en/PERSPECTRUM/dev.jsonl",
        "SCD": data_path + "en/SCD/dev.jsonl",
        "SemEval2019": data_path + "en/SemEval2019Task7/dev.jsonl",
        "Snopes": data_path + "en/Snopes/dev.jsonl"
    },
    "model": {
        "type": "multitask",
        "backbone": {
            "type": "sdmtl_backbone",
            "encoder": {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer",
                        "model_name": bert_model_name
                    }
                },
            },
        },
        "heads": {
            "SemEval2016": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
            "IBMCS": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 2,
                "dropout": dropout
            },
            "arc": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 4,
                "dropout": dropout
            },
            "ArgMin": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
            "FNC1": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 4,
                "dropout": dropout
            },
            "IAC": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
            "PERSPECTRUM": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
            "SCD": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
            "SemEval2019": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 4,
                "dropout": dropout
            },
            "Snopes": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 3,
                "dropout": dropout
            },
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01
        },
        "serialization_dir": output_dir,
        "validation_metric": [
            "+SemEval2016_f1_macro",
            "+IBMCS_f1_macro",
            "+arc_f1_macro",
            "+ArgMin_f1_macro",
            "+FNC1_f1_macro",
            "+IAC_f1_macro",
            "+PERSPECTRUM_f1_macro",
            "+SCD_f1_macro",
            "+SemEval2019_f1_macro",
            "+Snopes_f1_macro"
        ],
        "num_epochs": 5,
        "patience": 1,
        "cuda_device": cuda_device
    }
}
