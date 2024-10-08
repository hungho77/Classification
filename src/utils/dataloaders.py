import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torchvision.datasets as datasets
import torchvision.transforms as transforms


DATA_BACKEND_CHOICES = ['pytorch']

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    DATA_BACKEND_CHOICES.append('dali-gpu')
    DATA_BACKEND_CHOICES.append('dali-cpu')
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
    

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cou=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1
        
        self.input = ops.FileReader(
                    file_root = data_dir,
                    shard_id = local_rank,
                    num_shards = world_size,
                    random_shuffle = True)
        
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                          random_aspect_ratio=[0.75, 4./3],
                          random_area=[0.08, 1.0],
                          num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG 
            # buffers to be able to handle all images from full-sized
            # ImageNet without addingtional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.75, 4./3.],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)
        
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)
        
    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]
    
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = False)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = "gpu", resize_shorter = size)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

class DALIWrapper(object):
    def gen_wrapper(dalipipeline):
        for data in dalipipeline:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline):
        self.dalipipeline = dalipipeline

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline)
    
def get_dali_train_loader(dali_cpu=False):
    def gdtl(data_path, batch_size, workers=5, _worker_init_fn=None):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        traindir = os.path.join(data_path, 'train')

        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, crop = 224, dali_cpu=dali_cpu)

        pipe.build()
        test_run = pipe.run()
        train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(train_loader), int(pipe.epoch_size("Reader") / (world_size * batch_size))

    return gdtl

def get_dali_val_loader():
    def gdvl(data_path, batch_size, workers=5, _worker_init_fn=None):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        valdir = os.path.join(data_path, 'val')

        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir,
                crop = 224, size = 256)
        pipe.build()
        test_run = pipe.run()
        val_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size), fill_last_batch=False)

        return DALIWrapper(val_loader), int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl

def fast_collate(batch):
    imgs = [img['image'] for img in batch]
    targets = torch.tensor([target['path'] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield _input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            _input = next_input
            target = next_target

        yield _input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
#         if (self.dataloader.sampler is not None and
#             isinstance(self.dataloader.sampler,
#                        torch.utils.data.distributed.DistributedSampler)):

#             self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    
    
def find_classes(input_path):
    from pyarrow import parquet as pq
    df = pq.read_table(input_path)
    list_path = [i['path'] for i in df.select(['path']).to_pylist()]
    list_path = [os.path.basename(os.path.dirname(i)) for i in list_path]

    import numpy as np
    x = np.unique(list_path)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(x))}
    return class_to_idx

def get_pytorch_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224, num_epochs=1):
    from petastorm.pytorch import DataLoader
    from petastorm import make_reader, TransformSpec
    from cads_sdk.nosql import codec
    from PIL import Image
    
    class_to_idx = find_classes('{}_train.parquet'.format(data_path))
    
    def _transform_row(row):
        transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                ])
        return  {
            'image': transform(row['image']),
            'path': class_to_idx[os.path.basename(os.path.dirname(row['path']))]
        }

    
    transform = TransformSpec(_transform_row, removed_fields=['size'])
    
    reader = make_reader('{}_train.parquet'.format(data_path), 
                    reader_pool_type='dummy', num_epochs=num_epochs,
                    transform_spec=transform)
    nrows = 0
    for piece in reader.dataset.pieces:
        nrows += piece.get_metadata().num_rows
    dataset_len = int(round(nrows/batch_size))
        
    train_loader = DataLoader(reader, 
                    batch_size=batch_size,  
                    collate_fn=fast_collate,
                    shuffling_queue_capacity=int(round(nrows/3)))

    return PrefetchedWrapper(train_loader), dataset_len

def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    from cads_sdk.pytorch import DataLoader
    from petastorm import make_reader, TransformSpec
    from cads_sdk.nosql import codec
    from PIL import Image
    
    class_to_idx = find_classes('{}_val.parquet'.format(data_path))
    
    def _transform_row(row):
        transform = transforms.Compose([
                    transforms.Resize(input_size),
                    ])
        return  {
            'image': transform(row['image']),
            'path': class_to_idx[os.path.basename(os.path.dirname(row['path']))]
        }
    

    transform = TransformSpec(_transform_row, removed_fields=['size'])
    reader = make_reader('{}_val.parquet'.format(data_path), 
                    reader_pool_type='dummy',
                    transform_spec=transform)
    nrows = 0
    for piece in reader.dataset.pieces:
        nrows += piece.get_metadata().num_rows
    dataset_len = int(round(nrows/batch_size))
        
    val_loader = DataLoader(reader,
                batch_size=batch_size,  
                collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader), dataset_len
