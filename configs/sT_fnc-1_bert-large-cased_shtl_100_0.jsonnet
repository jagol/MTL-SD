local bert_model_name = "bert-large-cased";
local train_fname = "train.jsonl";
local dev_fname = "dev.jsonl";
local embedding_dim = 1024;
local dropout = 0.1;
local batch_size = 4;
local output_dir = "results/sT_FNC1_bert_large_shtl_100_0/";
local server = true;
local data_path = if server then "/srv/scratch0/jgoldz/mthesis/data/" else "/home/janis/Dropbox/UZH/UFSP_Digital_Religion/Master_Thesis/thesis_code/data/";
local cuda_device = 3;

local reader_common = {
        "max_sequence_length": 100,
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
            "FNC1": reader_common {
                "type": "FNC1"
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
        "FNC1": data_path + "fnc-1/" + train_fname
    },
    "validation_data_path": {
        "FNC1": data_path + "fnc-1/" + dev_fname
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
            "FNC1": {
                "type": "stance_head_two_layers",
                "input_dim": embedding_dim,
                "output_dim": 4,
                "dropout": dropout
            }
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
            "+FNC1_f1_macro"
        ],
        "num_epochs": 5,
        "patience": 1,
        "cuda_device": cuda_device
    }
}
