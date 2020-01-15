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
    self.single_GPU_batch_size = 32
    # This is considering multiple GPUs. Do NOT set this.
    self._batch_size = None
    # Do NOT set this.
    self._real_batch_size = None
    self.iter_num_update_optimizer = 1
    self.eval_gap_instance_num = None

    self.train_files = []
    self.vali_file = ""
    self.test_files = []

    self.train_sample_num = None

    self.incremental_train = True
    self.use_polynormial_decay = True
    self.warmup_ratio = 0.1

  # Have to invoke this function.
  def update(self):
    self._batch_size = max(1, len(self.gpus)) * self.single_GPU_batch_size
    self._real_batch_size = self._batch_size * self.iter_num_update_optimizer

  def verify(self):
    assert self.train_sample_num is not None
    assert self._batch_size is not None, "you have to call param.update()"
    assert self._real_batch_size is not None, "you have to call param.update()"
    assert self.iter_num_update_optimizer is not None
    assert self.eval_gap_instance_num is not None

    for file in self.train_files + self.test_files:
      for real_file in glob.glob(file):
        assert nlp.ensure_file_exist(real_file)

    Logger.info("\n", "-" * 64)
    for key in self.__dict__:
      Logger.info(f"{key:20}: {self.__dict__[key]}")
    Logger.info("-" * 64, "\n")
    


