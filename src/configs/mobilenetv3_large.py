from easydict import EasyDict as edict

config = edict()

# Network
config.output = "./trained_models/mobilenetv3_large_1.0_1"
config.network = "mobilenetv3_large"

# Hyper paramenters
config.width_mult = 1.0
config.resume = False
config.checkpoint = ""
config.pretrained = ""
config.weights = ""
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 0.4
config.lr_decay = 'cos'
config.gamma = 0.1
config.dali = False
config.evaluate = False
config.data_backend = 'pytorch'
config.distributed = False
config.seed = None
config.world_size = 0

# Wandb log
config.project = "facemaskclassify"
config.entity = "hungho7"
config.name = "mobilenetv3_large_1"

# Train
config.data_dir = "/home/hunght21/projects/insightface/alignment/facemask"
config.num_classes = 2
config.epochs = 150
config.warmup_epoch = 5
config.workers = 120
config.batch_size = 256
config.input_size = 224

