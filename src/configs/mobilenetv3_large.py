from easydict import EasyDict as edict

config = edict()

# Network
config.output = "./trained_models/mobilenetv3_large_1.0"
config.network = "mobilenetv3_large"

# Hyper paramenters
config.width_mult = 1.0
config.resume = False
config.checkpoint = ""
config.pretrained = "./pretrained/mobilenetv3_large_1.0.pth"
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
config.name = "mobilenetv3_large"

# Train
config.data_dir = "file:/home/hunght21/projects/facemaskclassification/data/facemask"
config.num_classes = 2
config.epochs = 150
config.warmup_epoch = 5
config.workers = 20
config.batch_size = 64
config.input_size = 224

