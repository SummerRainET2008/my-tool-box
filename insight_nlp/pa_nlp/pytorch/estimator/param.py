from pa_nlp import *
from pa_nlp import nlp
from pa_nlp.nlp import Logger

class ParamBase(abc.ABC):
  def __init__(self, model_name: str):
    self.model_name = model_name
    assert not nlp.is_none_or_empty(self.model_name)

    self.path_work  = f"work.{self.model_name}"
    nlp.ensure_folder_exists(self.path_work)

    self.path_data  = f"{self.path_work}/data"
    nlp.ensure_folder_exists(self.path_data)
    
    self.path_model = f"{self.path_work}/model"
    nlp.ensure_folder_exists(self.path_model)

    self.path_feat  = f"{self.path_work}/feat"
    nlp.ensure_folder_exists(self.path_feat)

    self.optimizer_name = "Adam"
    self.lr = 0.001
    self.lr_decay = 0.99
    self.lr_min = 0.0005
    self.l2 = 0
    self.param_norm = 5
    
    self.epoch_num = 1
    self.gpus = []
    self.batch_size_one_gpu = 32
    self.batch_size_inference_one_gpu = 128
    self.iter_num_update_optimizer = 1
    self.eval_gap_instance_num = None

    self.train_files = []
    self.vali_file = ""
    self.test_files = []

    self.train_sample_num = None

    self.incremental_train = False
    self.use_polynormial_decay = False
    self.warmup_ratio = 0   # 0.1

    self.model_kept_num = 3
    
  def size_divided_by_16(self, size, unit=16):   
    return math.ceil(size / unit) * unit
    
  def get_core_num(self):
    return max(1, len(self.gpus))
  
  def get_batch_size_all_gpus(self):
    return self.batch_size_one_gpu * self.get_core_num()
  
  def get_batch_size_per_optimization(self):
    return self.get_batch_size_all_gpus() * self.iter_num_update_optimizer
  
  def get_batch_size_inference_all_gpus(self):
    return self.batch_size_inference_one_gpu * self.get_core_num()

  def verify(self):
    assert self.train_sample_num is not None
    assert self.iter_num_update_optimizer is not None
    assert self.eval_gap_instance_num is not None

    for file in self.train_files + self.test_files:
      for real_file in glob.glob(file):
        assert nlp.ensure_file_exist(real_file)

    Logger.info("\n", "-" * 64)
    for key in sorted(self.__dict__):
      Logger.info(f"{key:20}: {self.__dict__[key]}")
      
    core_num = self.get_core_num()    
    Logger.info(
      f"batch_size[{core_num} GPUs]: {self.get_batch_size_all_gpus()}"
    )
    Logger.info(
      f"batch_size[{core_num} GPUs, {self.iter_num_update_optimizer} "
      f"gradient accumulations]: {self.get_batch_size_per_optimization()}"
    )
    Logger.info(
      f"batch_size inference[{core_num} GPUs]: "
      f"{self.get_batch_size_inference_all_gpus()}] "
    )
    Logger.info("-" * 64, "\n")
    
