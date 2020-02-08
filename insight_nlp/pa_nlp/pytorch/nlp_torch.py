#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import *
from pa_nlp.nlp import Logger
from pa_nlp import nlp
from pa_nlp.pytorch import *
import torch

def update_modules(module: nn.Module, module_block: typing.Any, name: str):
  if isinstance(module_block, nn.Module):
    module.add_module(name, module_block)

  elif isinstance(module_block, list):
    for idx, m in enumerate(module_block):
      update_modules(module, m, f"{name}_{idx}")

  elif isinstance(module_block, dict):
    for key, m in module_block.items():
      update_modules(module, m, f"{name}_{key}")

  else:
    assert False, type(module_block)

def sequence_mask(real_len: torch.Tensor,
                  max_size: typing.Union[int, None]=None):
  '''
  real_len: [batch]
  return: [batch, max_size]
  '''
  if max_size is None:
    max_size = real_len.max()

  size = torch.LongTensor(range(1, max_size + 1)).to(real_len)
  mask = real_len.unsqueeze(1) >= size.unsqueeze(0)

  return mask

def display_model_parameters(model: nn.Module):
  print("-" * 32)
  print(f"module parameters:")
  total_num = 0
  for name, var in model.named_parameters():
    print(name, var.shape)
    total_num += functools.reduce(operator.mul, var.shape, 1)
  print()
  print(f"#paramters: {total_num:_}")
  print("-" * 32)

class Swish(nn.Module):
  def __init__(self):
    super(Swish, self).__init__()

  def forward(self, x):
    return x * torch.sigmoid(x)

class Gelu(nn.Module):
  def __init__(self):
    super(Gelu, self).__init__()

  def forward(self, x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class FFN(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               output_dim,
               activation: nn.Module=Gelu(),
               dropout=0):
    super(FFN, self).__init__()
    self._layer = nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      activation,
      torch.nn.Dropout(dropout),
      torch.nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x: torch.Tensor):
    return self._layer(x)

class Dense(nn.Module):
  def __init__(self, linear_layer: nn.Linear, activation=nn.LeakyReLU()):
    super(Dense, self).__init__()
    if activation is None:
      self._layer = linear_layer
    else:
      self._layer = nn.Sequential(
        linear_layer, activation
      )

  def forward(self, x: torch.Tensor):
    return self._layer(x)

class Attention(nn.Module):
  def __init__(self,
               query_dim,
               values_dim,
               atten_dim):
    super(Attention, self).__init__()
    self._query_dense = Dense(nn.Linear(query_dim, atten_dim))
    self._values_dense = Dense(nn.Linear(values_dim, atten_dim))
    self._logit_out = Dense(nn.Linear(atten_dim, 1))

  def forward(self, query, values, mask):
    '''
    query: [batch, dim]
    values: [batch, len, dim]
    mask: [batch, max-seq], IntTensor
    '''
    m = self._values_dense(values)
    m += self._query_dense(query).unsqueeze(1).expand_as(m)
    logit = self._logit_out(torch.tanh(m)).squeeze(2)
    logit.data.masked_fill_(torch.logical_not(mask), -1e-9)
    prob = nn.functional.softmax(logit, dim=1)
    output = prob.unsqueeze(1).matmul(values).squeeze(1)

    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, dim, multihead=4):
    super(MultiHeadAttention, self).__init__()
    self._multihead_att = nn.MultiheadAttention(dim, multihead)

  def forward(self, query, values, key_mask=None):
    '''
    query: [seq-len, batch, dim]
    values: [seq-len, batch, dim]
    key_mask: [batch, seq-len]
    '''
    output, weights = self._multihead_att(
      query, values, values, key_padding_mask=key_mask
    )

    return output

class InnerAttention(nn.Module):
  def __init__(self, dim, atten_dim=512, multihead=4):
    super(InnerAttention, self).__init__()
    self._input_dense = Dense(nn.Linear(dim, atten_dim))
    self._multihead_att = MultiHeadAttention(atten_dim, multihead)
    self._query = nn.Parameter(torch.Tensor(1, 1, atten_dim))
    self._output_dense = Dense(nn.Linear(atten_dim, dim))

    self._init_weights()

  def _init_weights(self):
    # nn.init.xavier_uniform_(self._query)
    nn.init.uniform_(self._query, 0, 1)

  def forward(self, values, key_mask=None):
    '''
    values: [seq-len, batch, dim]
    key_mask: [batch, seq-len]
    '''
    values = self._input_dense(values)
    query = self._query.expand(1, values.size(1), -1)
    output = self._multihead_att(query, values, key_mask)
    output = self._output_dense(output)

    return output.squeeze(0)

class RNNEncoder(nn.Module):
  def __init__(self,
               layer_num,
               emb_dim,
               hidden_dim,
               out_dropout: float):
    super(RNNEncoder, self).__init__()
    self._bi_rnn = nn.GRU(
      input_size=emb_dim, hidden_size=hidden_dim, num_layers=layer_num,
      batch_first=True, bidirectional=True
    )

    self._combine_dense = Dense(nn.Linear(2 * hidden_dim, hidden_dim))
    self._out_dropout = nn.Dropout(out_dropout)
    self._layer_num = layer_num

    self._hidden_dim = hidden_dim

  def _init_hidden(self, batch_size):
    weight = next(self.parameters())
    hidden = weight.new_zeros(2 * self._layer_num, batch_size, self._hidden_dim)

    return hidden

  def forward(self, x, mask, packed_input=False):
    '''
    x: [batch, max-seq, emb-dim]
    mask: [batch, max-seq]
    '''
    hidden = self._init_hidden(x.size(0))
    x_length = mask.sum(1)

    #note, when pack_padded_sequence is needed, the batch has to been sorted by
    #their real lengths.
    if packed_input:
      x = torch.nn.utils.rnn.pack_padded_sequence(
        x, x_length, batch_first=True
      )
    output, _ = self._bi_rnn(x, hidden)
    if packed_input:
      output, _ = torch.nn.utils.rnn.pad_packed_sequence(
        output, batch_first=True, total_length=mask.size(1)
      )

    output = self._combine_dense(output)
    output = self._out_dropout(output)

    return output

# google's style
class RNNEncoder1(nn.Module):
  def __init__(self,
               layer_num,
               emb_dim,
               hidden_dim,
               out_dropout: float):
    super(RNNEncoder1, self).__init__()

    self._input_dense = Dense(nn.Linear(emb_dim, hidden_dim))

    for layer_id in range(layer_num):
      if layer_id == 0:
        # we assume GRU works correctly in the bidirectional style.
        layer = nn.GRU(
          input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
          batch_first=True, bidirectional=True
        )
      else:
        layer = nn.GRU(
          input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1,
          batch_first=True, bidirectional=False
        )

      self.add_module(f"_layer_{layer_id}", layer)

    self._combine_dense = Dense(nn.Linear(2 * hidden_dim, hidden_dim))
    self._out_dropout = nn.Dropout(out_dropout)

    self._layer_num = layer_num
    self._hidden_dim = hidden_dim

  def _init_hidden(self, batch_size):
    weight = next(self.parameters())
    hiddens = []
    hiddens.append(weight.new_zeros(2, batch_size, self._hidden_dim))
    for _ in range(1, self._layer_num):
      hiddens.append(weight.new_zeros(1, batch_size, self._hidden_dim))

    return hiddens

  def forward(self, x, mask, packed_input=False):
    '''
    x: [batch, max-seq, emb-dim]
    mask: [batch, max-seq]
    '''
    hiddens = self._init_hidden(x.size(0))
    x = self._input_dense(x)
    for layer_id in range(self._layer_num):
      layer = getattr(self, f"_layer_{layer_id}")
      input = x
      x, _ = layer(x, hiddens[layer_id])
      if layer_id == 0:
        x = self._combine_dense(x)
      x += input

    x = self._out_dropout(x)

    return x

class ResidualGRU(nn.Module):
  def __init__(self, hidden_size, dropout=0.1, num_layers=2):
    super(ResidualGRU, self).__init__()
    self._enc_layer = nn.GRU(
      input_size=hidden_size, hidden_size=hidden_size // 2,
      num_layers=num_layers, batch_first=True, dropout=dropout,
      bidirectional=True
    )
    self._out_norm = nn.LayerNorm(hidden_size)

  def forward(self, input):
    output, _ = self._enc_layer(input)
    return self._out_norm(output + input)

class VallinaDecoder(nn.Module):
  def __init__(self,
               emb_dim,
               hidden_dim,
               enc_outputs_dim,
               out_dropout):
    super(VallinaDecoder, self).__init__()

    self._rnn_cell1 = nn.GRUCell(emb_dim, hidden_dim)
    self._enc_attn = Attention(hidden_dim, enc_outputs_dim, enc_outputs_dim)
    self._rnn_cell2 = nn.GRUCell(enc_outputs_dim, hidden_dim)
    self._x_dense = Dense(nn.Linear(emb_dim, hidden_dim))

    self._out_dropout = nn.Dropout(out_dropout)

  def forward(self, x, hidden, enc_outputs, enc_mask):
    '''
    x: [batch, emb-dim]
    hidden: [batch, hidden-dim]
    x_mask: [batch, max-seq]
    enc_output: [batch, max-seq, dim]
    '''
    hidden = self._rnn_cell1(x, hidden)
    attn_enc = self._enc_attn(hidden, enc_outputs, enc_mask)
    hidden = self._rnn_cell2(attn_enc, hidden)
    output = torch.tanh(
      self._x_dense(x) + attn_enc + hidden
    )
    output = self._out_dropout(output)

    return output, hidden

class TextCNN(nn.Module):
  def __init__(self,
               kernels: typing.List[int],
               in_channel: int,
               out_channel: int,
               max_seq_len: int, dim: int,
               activation=nn.LeakyReLU(),
               dropout=0):
    super(TextCNN, self).__init__()

    self._cnns = [
      nn.Sequential(
        nn.Conv2d(in_channel, out_channel, (kernel, dim)),
        activation,
        nn.MaxPool2d((max_seq_len - kernel + 1, 1)),
      )
      for kernel in kernels
    ]
    update_modules(self, self._cnns, "_cnns")
    self._out_dropout = nn.Dropout(dropout)
    self._output_size = len(kernels) * out_channel

    self._init_weights()

  def _init_weights(self):
    for name, w in self.named_parameters():
      if "cnn" in name and "weight" in name:
        nn.init.normal_(w, 0, 0.1)
      elif "bias" in name:
        nn.init.zeros_(w)

  def forward(self, x):
    '''
    x: [batch, channel, word_num, dim]
    '''

    outs = [cnn(x) for cnn in self._cnns]
    out = torch.cat(outs, -1)
    out = out.flatten(1, -1)
    assert out.shape[-1] == self._output_size, \
      f"{out.shape} != {self._output_size}"
    out = self._out_dropout(out)

    return out

