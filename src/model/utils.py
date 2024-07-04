import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from logging import getLogger
from .bucketed_embedding import BucketedEmbedding


logger = getLogger()

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


def get_recurrent_module(module_type):
    #得到循环神经网络类型
    if module_type == 'rnn':
        return nn.RNN
    elif module_type == 'gru':
        return nn.GRU
    elif module_type == 'lstm':
        return nn.LSTM
    else:
        raise Exception("Unknown recurrent module type: '%s'" % module_type)


def value_loss(delta):
    """
    MSE Loss / Smooth L1 Loss / Huber Loss
    DQN部分的loss计算
    已被弃用
    """
    assert delta >= 0
    if delta == 0:
        # MSE Loss
        return nn.MSELoss()
    elif delta == 1:
        # Smooth L1 Loss
        return nn.SmoothL1Loss()
    else:
        # Huber Loss
        def loss_fn(input, target):
            diff = (input - target).abs()
            diff_delta = diff.cmin(delta)
            loss = diff_delta * (diff - diff_delta / 2)
            return loss.mean()
        return loss_fn



def build_CNN_network(module, params):
    """
    Build CNN network.
    """
    # model parameters
    module.hidden_dim = params.hidden_dim
    module.dropout = params.dropout
    module.n_actions = params.n_actions

    # screen input format - for RNN, we only take one frame at each time step
    if hasattr(params, 'recurrence') and params.recurrence != '':
        in_channels = params.n_fm
    else:
        in_channels = params.n_fm * params.hist_size
    height = params.height
    width = params.width
    logger.info('Input shape: %s' % str((params.n_fm, height, width)))

    # convolutional layers
    module.conv = nn.Sequential(*filter(bool,[
        nn.Conv2d(in_channels, 32, (8, 8), stride=(4, 4)),
        None if not params.use_bn else nn.BatchNorm2d(32),
        nn.ReLU(),
        
        nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
        None if not params.use_bn else nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # 三层卷积
        nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
        None if not params.use_bn else nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # 去除dropout
    ]))

    # get the size of the convolution network output得到卷积层输出尺寸
    x = Variable(torch.FloatTensor(1, in_channels, height, width).zero_())
    module.conv_output_dim = module.conv(x).nelement()


def build_game_variables_network(module, params):
    """
    Build game variables network (health, ammo, etc.)
    """
    module.game_variables = params.game_variables
    module.n_variables = params.n_variables
    module.game_variable_embeddings = []
    for i, (name, n_values) in enumerate(params.game_variables):
        embeddings = BucketedEmbedding(params.bucket_size[i], n_values,
                                       params.variable_dim[i])
        setattr(module, '%s_emb' % name, embeddings)
        module.game_variable_embeddings.append(embeddings)


def build_game_features_network(module, params):
    """
    Build game features network.
    建造游戏特征网络
    """
    module.game_features = params.game_features
    if module.game_features:
        module.n_features = module.game_features.count(',') + 1
        module.proj_game_features = nn.Sequential(
            nn.Dropout(module.dropout),
            nn.Linear(module.conv_output_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(module.dropout),
            nn.Linear(params.hidden_dim, module.n_features),
            nn.Sigmoid(),
        )
    else:
        module.n_features = 0
