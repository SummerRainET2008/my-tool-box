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

    self._global_batch_id = 0
    self._opt_vali_error = 0
    self._run_sample_num = 0
    self._last_vali_sample_num = 0

    if param.use_polynormial_decay:
      assert param.train_sample_num is not None
      total_step = param.train_sample_num * param.epoch_num // param.batch_size
      if param.use_warmup:
        total_step -= param.warmup_steps
      assert total_step > 0

      # self._lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(
      #   param.lr, total_step, end_learning_rate=0.,
      # )

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

  def _get_learning_rate(self):
    #print(f"get_learning_rate")
    '''param = self._param
    lr = param.lr
    global_step = self._checkpoint.global_batch_step
    if param.use_polynormial_decay:
      if param.use_warmup:
        lr = self._lr_decay(global_step - param.warmup_steps)
      else:
        lr = self._lr_decay(global_step)

    if param.use_warmup:
      assert param.warmup_steps is not None

      global_steps_int = nlp_tf.to_int(global_step)
      warmup_steps_int = tf.constant(param.warmup_steps, dtype=tf.int32)

      global_steps_float = nlp_tf.to_double(global_steps_int)
      warmup_steps_float = nlp_tf.to_double(warmup_steps_int)

      warmup_percent_done = global_steps_float / warmup_steps_float
      warmup_lr = param.lr * warmup_percent_done

      is_warmup = nlp_tf.to_double(global_steps_int < warmup_steps_int)
      lr = (1.0 - is_warmup) * lr + is_warmup * warmup_lr'''

    return self._param.lr

  def load_model(self):
    param = self._param
    try:
      check_point_file = f"{param.path_model}/checkpoint"
      if not os.path.isfile(check_point_file):
        Logger.info("No model to load")
        return

      g_batch_id = open(check_point_file).readlines()[-1]
      model_file = f"{param.path_model}/model_{g_batch_id}.pt"
      checked_data = torch.load(model_file)

      self._global_batch_id = checked_data[0]
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
    name = f'model_{self._global_batch_id}.pt'
    nlp.execute_cmd(
      f"echo {self._global_batch_id} >> {param.path_model}/checkpoint"
    )

    torch.save(
      [
        self._global_batch_id,
        self._opt_vali_error,
        self._run_sample_num,
        self._model.state_dict()
      ],
      os.path.join(param.path_model, name)
    )


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

  def train(self):
    batch_num, total_loss = 0, 0.
    while True:
      try:
        start_time = time.time()
        epoch_id, batch  = next(self._train_data_iter)
        duration = time.time() - start_time
        Logger.debug(f"batch data fetch time: {duration} sec.")
      except StopIteration:
        break

      start_time = time.time()
      self._model.train()
      batch = [e.to(self._device) for e in batch]
      batch_loss = self._train_one_batch(*batch)
      duration = time.time() - start_time

      batch_num += 1
      total_loss += batch_loss
      self._run_sample_num += batch[0].size(0)
      self._global_batch_id += 1
      Logger.info(
        f"Epoch: {epoch_id} batch: {self._global_batch_id} "
        f"samples: {self._run_sample_num} "
        f"loss: {batch_loss} time: {duration:.4f} "
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
    if diff >= self._param.evaluate_freq:
      self._last_vali_sample_num = self._run_sample_num
      return True

    return False

