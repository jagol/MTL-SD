local bert_model_name = "bert-base-cased";
local train_fname = "train.jsonl";
local embedding_dim = 768;
local dropout = 0.1;
local batch_size = 8;
local output_dir = "results/base_conf/";
local server = false;
local data_path = if server then "/srv/scratch0/jgoldz/mthesis/data/" else "/home/janis/Dropbox/UZH/UFSP_Digital_Religion/Master_Thesis/thesis_code/data/";
local cuda_device = -1;

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
            "SemEval2016": reader_common {
                "type": "SemEval2016"
            }
//            "IBMCS": reader_common {
//                "type": "IBMCS"
//            },
//            "arc": reader_common {
//                "type": "arc"
//            }
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
         "SemEval2016": data_path + "en/SemEval2016Task6/" + train_fname,
//         "IBMCS": data_path + "en/IBM_CLAIM_STANCE/" + train_fname,
//        "arc": data_path + "en/arc/" + train_fname
    },
    "validation_data_path": {
         "SemEval2016": data_path + "en/SemEval2016Task6/dev.jsonl",
//         "IBMCS": data_path + "en/IBM_CLAIM_STANCE/dev.jsonl",
//        "arc": data_path + "en/arc/dev.jsonl"
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
            }
//            "IBMCS": {
//                "type": "stance_head_two_layers",
//                "input_dim": embedding_dim,
//                "output_dim": 2,
//                "dropout": dropout
//            },
//            "arc": {
//                "type": "stance_head_two_layers",
//                "input_dim": embedding_dim,
//                "output_dim": 4,
//                "dropout": dropout,
//            },
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
        "validation_metric": ["+SemEval2016_f1_macro"], // ["+", "+IBMCS_f1_macro", "+arc_f1_macro"],
        "num_epochs": 1,
        "patience": 2,
        "cuda_device": cuda_device
    }
}
