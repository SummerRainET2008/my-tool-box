#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.pytorch.estimator.param import ParamBase
import torch
from pa_nlp.pytorch import *
from torch.optim import Optimizer

class TrainerBase(abc.ABC):
  def __init__(self, param: ParamBase, model,
               train_data_iter, optimizer: typing.Union[Optimizer, None]=None):
    gpu_ids = param.gpus
    assert isinstance(gpu_ids, list)
    if len(gpu_ids) == 0:
      self._device = torch.device("cpu")
    else:
      self._device = torch.device(f"cuda:{gpu_ids[0]}")
      model = nn.DataParallel(
        model, device_ids=[f"cuda:{gid}" for gid in gpu_ids]
      )
    model = model.to(self._device)

    if not param.incremental_train:
      nlp.ensure_folder_exists(param.path_model, True)

    self._global_step_id = 0
    self._opt_vali_error = 0
    self._run_sample_num = 0
    self._last_vali_sample_num = 0

    if optimizer is not None:
      self._optimizer = optimizer
    else:
      self._optimizer = getattr(torch.optim, param.optimizer_name)(
        model.parameters(), lr=param.lr, weight_decay=param.l2
      )

    self._param = param
    self._train_data_iter = train_data_iter
    self._model = model

    self.load_model()
    
  def _step_optimizer(self, buff={}):
    param_groups = self._optimizer.param_groups 
    if "lr" not in buff:
      buff["lr"] = [group["lr"] for group in param_groups]

    lr_ratio = self._get_lr_ratio()
    for group, lr in zip(param_groups, buff["lr"]):
      group["lr"] = lr * lr_ratio
    Logger.info(f"learning rates:", [g["lr"] for g in param_groups])   
      
    self._optimizer.step()   

  def _get_lr_ratio(self, buff={}):
    '''
    As there may have different learning rates, we only return the ratio of
    current learning rate divided by the designated learning rate. 
    '''
    param = self._param
    total_steps = math.ceil(
      param.train_sample_num * param.epoch_num / param._batch_size_per_opt
    )
    warmup_steps = max(1, math.ceil(param.warmup_ratio *  total_steps))
    if  param.use_polynormial_decay:
      decay_total_steps = total_steps
    else:
      decay_total_steps = nlp.INF

    ratio1 = (self._global_step_id + 1) / warmup_steps
    ratio2 = 1 - (self._global_step_id + 1 - warmup_steps) / \
             (decay_total_steps - warmup_steps)

    return min(ratio1, ratio2)

  def load_model(self):
    param = self._param
    try:
      check_point_file = f"{param.path_model}/checkpoint"
      if not os.path.isfile(check_point_file):
        Logger.info("No model to load")
        return

      model_name = open(check_point_file).readlines()[-1].strip()
      model_file = f"{param.path_model}/{model_name}"
      checked_data = torch.load(model_file)

      self._global_step_id = checked_data[0]
      self._opt_vali_error = checked_data[1]
      self._run_sample_num = checked_data[2]
      state_dict = checked_data[3]
      self._model.load_state_dict(state_dict)
      self._last_vali_sample_num = self._run_sample_num
      Logger.info(f"Model load succeeds: {model_file}")

    except Exception as error:
      Logger.info(f"Model load fails: {error}")

  @abc.abstractmethod
  def evaluate_file(self, data_file) -> float:
    '''return a float denoting its error. Smaller, better.'''
    pass

  @abc.abstractmethod
  def predict(self, batch_data):
    pass

  def save_model(self):
    param = self._param
    name = f'model_{self._global_step_id}.pt'
    nlp.execute_cmd(
      f"echo {name} >> {param.path_model}/checkpoint"
    )

    torch.save(
      [
        self._global_step_id,
        self._opt_vali_error,
        self._run_sample_num,
        self._model.state_dict()
      ],
      os.path.join(param.path_model, name)
    )

    model_names = open(f"{param.path_model}/checkpoint").read().split()
    for name in model_names[: -param.model_kept_num]:
      model_file = f"{param.path_model}/{name}"
      if os.path.isfile(model_file):
        nlp.execute_cmd(f"rm {model_file}")

  def _try_to_save_best_model(self):
    if nlp.is_none_or_empty(self._param.vali_file):
      self.save_model()

    else:
      eval_error = self.evaluate_file(self._param.vali_file)
      if eval_error < self._opt_vali_error:
        self._opt_vali_error = eval_error
        self.save_model()

  def _evaluate(self):
    self._try_to_save_best_model()

    for data_file in self._param.test_files:
      self.evaluate_file(data_file)

  @abc.abstractmethod
  def _train_one_batch(self, *batch)-> float:
    pass

  def _get_batch_data(self):
    while True:
      try:
        start_time = time.time()
        epoch_id, batch  = next(self._train_data_iter)
        batch = [e.to(self._device) for e in batch]
        duration = time.time() - start_time
        Logger.debug(f"batch data fetch time: {duration} sec.")
        yield epoch_id, batch

      except StopIteration:
        break

  def train(self):
    batch_num, total_loss = 0, 0.
    train_start_time = time.time()
    batch_iter = self._get_batch_data()

    while True:
      start_time = time.time()
      self._model.train()
      self._optimizer.zero_grad()
      batch_loss = []

      for _, [epoch_id, batch] in zip(
        range(self._param.iter_num_update_optimizer), batch_iter
      ):
        single_batch_loss = self._train_one_batch(*batch)
        single_batch_loss.backward()
        batch_loss.append(single_batch_loss.cpu().detach().numpy())
        self._run_sample_num += batch[0].size(0)
        
      if len(batch_loss) == 0:
        break

      torch.nn.utils.clip_grad_norm_(
        self._model.parameters(), self._param.param_norm
      )
      self._step_optimizer()
      batch_loss = sum(batch_loss) / len(batch_loss)
      duration = time.time() - start_time

      batch_num += 1
      total_loss += batch_loss
      self._global_step_id += 1
      train_duration = time.time() - train_start_time
      Logger.info(
        f"Epoch: {epoch_id} batch: {self._global_step_id} "
        f"samples: {self._run_sample_num} "
        f"loss: {batch_loss} time: {duration:.4f} "
        f"training time: {nlp.to_readable_time(train_duration)} "
      )

      if batch_num % 5 == 0:
        avg_loss = total_loss / batch_num
        Logger.info(f"avg_loss[{batch_num}]: {avg_loss:.4f}")
        batch_num, total_loss = 0, 0.

      if self._when_evaluate():
        self._model.eval()
        self._evaluate()

  def _when_evaluate(self):
    diff = self._run_sample_num - self._last_vali_sample_num
    if diff >= self._param.eval_gap_instance_num:
      self._last_vali_sample_num = self._run_sample_num
      return True

    return False

