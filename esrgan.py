from typing import Tuple, Optional, Dict, List, Union, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
import pytorch_lightning as pl

def norm_zero_255(img: torch.Tensor) -> torch.Tensor:
  """This function takes as input an image in torch.Tensor format with values between [0,1] 
      and normalize it between [0,255].

  Args:
    - img (torch.Tensor): Image with values between [0,1] that has to be normalized between [0,255].

  Returns:
    - torch.Tensor: Normalized image between [0,255].
  """
  img = torch.mul(img,255.0)
  img = torch.round(img).float()
  return img

class ResBlock(nn.Module):
  """Residual Dense Block.
  Args:
    - in_channels (int): The number of input channels for each conv in the residual dense block.
    - out_channels (int): The number of output channels for each conv in the residual dense block.
    - beta (float): Residual scaling parameter, it's used in order to facilitate the training of a deep network. 
                    It scales down the residuals before adding them to the main path to prevent instability.

  Attributes:
    - conv1 (nn.Conv2d): First convolution.
    - conv2 (nn.Conv2d): Second convolution.
    - conv3 (nn.Conv2d): Third convolution.
    - conv4 (nn.Conv2d): Fourth convolution.
    - conv5 (nn.Conv2d): Fifth convolution.
    - leaky_relu (nn.LeakyReLU): A nn.LeakyReLU(0.2, True) applied after each convolution beside the fifth one. 
  """
  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               beta: float = 0.2) -> None:
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels + out_channels * 0, out_channels,kernel_size=3,stride=1,padding=1)
    self.conv2 = nn.Conv2d(in_channels + out_channels * 1, out_channels,kernel_size=3,stride=1,padding=1)
    self.conv3 = nn.Conv2d(in_channels + out_channels * 2, out_channels,kernel_size=3,stride=1,padding=1)
    self.conv4 = nn.Conv2d(in_channels + out_channels * 3, out_channels,kernel_size=3,stride=1,padding=1)
    self.conv5 = nn.Conv2d(in_channels + out_channels * 4, in_channels,kernel_size=3,stride=1,padding=1)
    self.leaky_relu = nn.LeakyReLU(0.2, True)
    self.beta = beta
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.

    Args:
      -  x (torch.Tensor): Input torch.Tensor.

    Returns:
      - torch.Tensor : Output torch.Tensor.
    """
    identity = x
    x1 = self.leaky_relu(self.conv1(x))
    x2 = self.leaky_relu(self.conv2(torch.cat([x,x1],1)))
    x3 = self.leaky_relu(self.conv3(torch.cat([x,x1,x2],1)))
    x4 = self.leaky_relu(self.conv4(torch.cat([x,x1,x2,x3],1)))
    x5 = self.conv5(torch.cat([x,x1,x2,x3,x4],1))
    x6 = torch.mul(x5, self.beta)
    x = torch.add(x6,identity)
    return x

class ResInResDenseBlock(nn.Module):
  """Residual in Residual Dense Block.

  Args:
    - in_channels (int): The number of input channels for each conv in the residual dense block.
    - out_channels (int): The number of output channels for each conv in the residual dense block.
    - beta (float): Residual scaling parameter, it's used in order to facilitate the training of a deep network. 
                    It scales down the residuals before adding them to the main path to prevent instability.
  
  Attributes:
    - resBlock3 (ResBlock): The first Residual Dense Block.
    - resBlock1 (ResBlock): The second Residual Dense Block.
    - resBlock2 (ResBlock): The third Residual Dense Block.
  """
  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               beta: float = 0.2) -> None:
    super().__init__()
    self.beta = beta
    self.resBlock1 = ResBlock(in_channels,out_channels,beta)
    self.resBlock2 = ResBlock(in_channels,out_channels,beta)
    self.resBlock3 = ResBlock(in_channels,out_channels,beta)
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.
    
    Args:
      -  x (torch.Tensor): Input torch.Tensor.

    Returns:
      - torch.Tensor : Output torch.Tensor.
    """
    identity = x
    x = self.resBlock1(x)
    x = self.resBlock2(x)
    x = self.resBlock3(x)
    x = torch.mul(x, self.beta)
    x = torch.add(x,identity)
    return x

class UpscaleBlock(nn.Module):
  """Upscale-Block used in order to scale up the output of the residual in residual dense block.

  Args:
    - in_channels (int): The number of input channels for each conv used before the pixelshuffle.
    - out_channels (int): The number of output channels for each conv used before the pixelshuffle.
    - scale (int): How much to upscale.

  Attributes:
    - conv1 (nn.Conv2d): The first convolution.
    - conv2 (nn.Conv2d): The second convolution.
    - shuffle1 (nn.PixelShuffle): The nn.PixelShuffle(2) to apply after conv1.
    - shuffle2 (nn.PixelShuffle): The nn.PixelShuffle(2) to apply after conv2.
    - leakyRelu (self.leakyRelu): A nn.LeakyReLU(0.2, True) to apply after the shuffles.
  """
  def __init__(self, 
               in_channels: int, 
               out_channels: int,
               scale: int) -> None:
    super().__init__()

    self.scale = scale
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)    
    self.shuffle1 = nn.PixelShuffle(2)
    
    if scale == 4:
      self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
      self.shuffle2 = nn.PixelShuffle(2)

    self.leakyRelu = nn.LeakyReLU(0.2, True)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """The forward pass.
    
    Args:
      -  x (torch.Tensor): Input torch.Tensor.

    Returns:
      - torch.Tensor : Output torch.Tensor.
    """
    x = self.leakyRelu(self.shuffle1(self.conv1(x)))
    if self.scale == 4:
      x = self.leakyRelu(self.shuffle2(self.conv2(x)))
    return x

class Generator(pl.LightningModule):
  """Generator used in the GAN for generating the fake_images.

  Args:
    - in_channels (int): The number of input channels for each conv used before the pixelshuffle.
    - out_channels (int): The number of output channels for each conv used before the pixelshuffle.
    - numBlocks (int): The number of residual in residual dense block used.
    - beta (float): Residual scaling parameter, it's used in order to facilitate the training of a deep network. 
                    It scales down the residuals before adding them to the main path to prevent instability.

  Arguments:
    - example_input_array (torch.Tensor): An example of input data.
    - preResConv (nn.Conv2d): Pre Residual Blocks convolution.
    - RRDBs (nn.ModuleList()): A nn.ModuleList() of ResInResDenseBlock defined using the makeResLayers() function.
    - postResConv (nn.Conv2d): Post Residual Blocks convolution.
    - upscale (UpscaleBlock): Upscale-Block used in order to scale up the output of the residual in residual dense block.
    - finalConv1 (nn.Conv2d): The first final convolution.
    - finalConv2 (nn.Conv2d): The second final convolution.
    - leakyRelu (nn.LeakyReLU): A nn.LeakyReLU(0.2, True) to apply between the final convolutions.
 """
  def __init__(self,
               in_channels: int,
               out_channels: int,
               numBlocks: int,
               beta: float,
               lr: float = 2e-4):
    super().__init__()

    self.beta = beta
    self.numBlocks = numBlocks
    self.lr = lr
    
    self.example_input_array = torch.rand(1, 3, 32, 32)

    self.preResConv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1, padding=1)
    self.RRDBs = self.makeResLayers(numBlocks, 64, 32)
    self.postResConv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    self.upscale = UpscaleBlock(out_channels, out_channels*4, scale=4)
    
    self.finalConv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.finalConv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
    self.leakyRelu = nn.LeakyReLU(0.2, True)

  def makeResLayers(self, 
                    numBlocks: int, 
                    in_channels: int, 
                    out_channels: int) -> nn.ModuleList():
    """Function used to make the Residual Layers.

    Args:
      - numBlocks (int): Number of ResInResDenseBlock to append to the nn.ModuleList().
      - in_channels (int): Input channels of each ResInResDenseBlock.
      - out_channels (int): Output channels of each ResInResDenseBlock.

    Returns:
      - nn.ModuleList
    """
    blocks = nn.ModuleList()
    for i in range(self.numBlocks):
      blocks.append(ResInResDenseBlock(in_channels,out_channels,self.beta))
    return blocks

  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.

    Args:
      - x (torch.Tensor): The input tensor.

    Returns:
      - torch.Tensor : The output tensor.
    """
    x = self.preResConv(x)
    identity = x

    for i in range(self.numBlocks):
      x = self.RRDBs[i](x)
    
    x = self.postResConv(x)
    x += identity
    x = self.upscale(x)
    
    x = self.leakyRelu(self.finalConv1(x))
    x = self.finalConv2(x)
    x = torch.clamp_(x, min=0, max=1)

    return x

  def training_step(self,
                    batch: Tuple[torch.Tensor, torch.Tensor],
                    batch_idx: int) -> float: 
    """ A very simple training loop.

    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch idx.

    Returns:
      - loss (float): The training loss.
    """
    x, y = batch
    y_hat = self(x)

    criterion = nn.L1Loss()
    
    loss = criterion(y_hat,y)
    self.log('l1_loss', loss, on_step=True, prog_bar=True)

    return loss

  def validation_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> float:
    """ A very simple validation loop.

    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch index.
    
    Returns:
      - val_l1_loss (float): The validation loss.
    """
    x, y = batch
    y_hat = self(x)

    criterion = nn.L1Loss()

    val_l1_loss = criterion(y_hat, y)
    self.log('val_l1_loss', val_l1_loss, on_step=True, prog_bar=True)
    
    return val_l1_loss

  def test_step(self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                batch_idx: int) -> float:
    """ A very simple test loop.
    
    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch index.
    
    Returns:
      - psnr (float): The test loss.
    """
    x, y = batch
    y_hat = norm_zero_255(self(x).detach().clone())
    y = norm_zero_255(y.detach().clone())
    psnr = PSNR(y_hat, y)
    self.log('test_psnr', psnr, on_step=True, prog_bar=True)
    return psnr
    
  def configure_optimizers(self) -> torch.optim.Optimizer:
    """Configure the optimizer.

    Returns:
      - optimizer (torch.optim.Optimizer): The chosen optimizer.
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

class DiscriminatorBlock(nn.Module):
  """Discriminator fundamental block.

  Args:
    - in_channels (int): Number of input channels of conv1.
    - out_channels (int): Number of output channels of conv1.
    - stride (int): The stride of the conv1.
    - kernel (int): The kernel for conv1.

  Attributes:
    - conv1 (nn.Conv2d): A convolution used to condence the images information. 
    - bn1 (nn.BatchNorm2d): A nn.BatchNorm2d(out_channels) performed after conv1.
    - leaky_relu (nn.LeakyReLU): A nn.LeakyReLU(0.2,True) performed after bn1.
  """
  def __init__(self,
               in_channels: int,
               out_channels: int,
               stride: int,
               kernel: int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.leaky_relu = nn.LeakyReLU(0.2,True)
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.
    
    Args:
      -  x (torch.Tensor): Input torch.Tensor.

    Returns:
      - torch.Tensor : Output torch.Tensor.
    """
    x = self.leaky_relu(self.bn1(self.conv1(x)))
    return x

class Discriminator(nn.Module):
  """The Discriminator Module.

  Attributes:
    - conv1 (nn.Conv2d): The first convolution for feature extraction. We have used nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True).
    - blocks (nn.ModuleList): Our Module List of Discriminator Block.
    - dense1 (nn.Linear): The first dense layer in which we defines our hidden units (neurons). We have used nn.Linear(512*4*4, 100).
    - dense2 (nn.Linear): The second dense layer which is our output layer. We have used nn.Linear(100, 1).
    - leaky_relu (nn.LeakyReLU): The activation layer between the first and the second layer. We have used nn.LeakyReLU(0.2, True).
    - example_input_array (torch.Tensor): An example of input array, torch.rand(1, 3, 128, 128).
  """
  def __init__(self):
      super().__init__()

      self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
      self.blocks = self.makeBlockLayers(3, 64)
      self.dense1 = nn.Linear(512*4*4, 100)
      self.dense2 = nn.Linear(100, 1)
      self.leaky_relu = nn.LeakyReLU(0.2, True)
      self.example_input_array = torch.rand(1, 3, 128, 128)

  def makeBlockLayers(self,
                      in_channels: int,
                      out_channels: int) -> nn.ModuleList:
    """The fucntion used for generating a nn.ModuleList containing our Discriminator Blocks.

    Args:
      - in_channels (int): Number of input channels.
      - out_channels (int): Number of output channels.

    Returns:
      - nn.ModuleList : A Module List containing our Discriminator Blocks.
    """
    blocks = nn.ModuleList()
    multiply_factor = 2
    for i in range(1,6):
      if i == 5:
        blocks.append(DiscriminatorBlock(in_channels,in_channels,stride=1,kernel=3))
        blocks.append(DiscriminatorBlock(in_channels,in_channels,stride=2,kernel=4))
        break
      if i != 1:
        blocks.append(DiscriminatorBlock(in_channels,out_channels,stride=1,kernel=3))
      blocks.append(DiscriminatorBlock(out_channels,out_channels,stride=2,kernel=4))
      in_channels = out_channels
      out_channels = out_channels*multiply_factor
    return blocks

  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.

    Args:
      - x (torch.Tensor): The input torch.Tensor.

    Returns:
      - torch.Tensor : the output torch.Tensor.
    """
    x = self.leaky_relu(self.conv1(x))
    
    for i in range(9):
      x = self.blocks[i](x)
    
    x = torch.flatten(x,start_dim=1)

    x = self.leaky_relu(self.dense1(x))
    x = self.dense2(x)

    return x

class ContentLoss(nn.Module):
  """

  Args:
    - feature_model_extractor_node (str): Layer of the VGG19 to be extracted. 
    - feature_model_normalize_mean (List): Mean applied as pre-processing step to the input images of the VGG19.
    - feature_model_normalize_std (List): Standard Deviation applied as pre-processing step to the input images of the VGG19.

  Arguments:
    - feature_extractor (torch.fx.graph_module.GraphModule):
    - normalize (transforms.Normalize): Applying the normalization to feature_model_normalize_mean and feature_model_normalize_std
  """
  def __init__(self,
               feature_model_extractor_node: str,
               feature_model_normalize_mean: List,
               feature_model_normalize_std: List) -> None:
    super(ContentLoss, self).__init__()
    # Get the name of the specified feature extraction node
    self.feature_model_extractor_node = feature_model_extractor_node
    # Load the VGG19 model trained on the ImageNet dataset.
    model = models.vgg19(True)
    # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
    self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
    # set to validation mode
    self.feature_extractor.eval()

    # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
    self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

    # Freeze model parameters.
    for model_parameters in self.feature_extractor.parameters():
        model_parameters.requires_grad = False

  def forward(self,
              sr_tensor: torch.Tensor,
              hr_tensor: torch.Tensor) -> torch.Tensor:
    """The forward pass.

    Args:
      - sr_tensor (torch.Tensor): Small resolution tensor.
      - hr_tensor (torch.Tensor): High resolution tensor.

    Returns:
      - content_loss (torch.Tensor): The content loss.
    """
    # Standardized operations
    sr_tensor = self.normalize(sr_tensor)
    hr_tensor = self.normalize(hr_tensor)

    sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
    hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

    # Find the feature map difference between the two images
    content_loss = F.l1_loss(sr_feature, hr_feature)

    return content_loss

class GAN(pl.LightningModule):
  """The GAN module define as a pl.LightningModule.

  Args:
    - generator (pl.LightningModule): The generator Module.
    - lr_generator (float): The learning rate of the generator. Default: 1e-4.
    - lr_discriminator (float): The learning rate of the discriminator. Default: 1e-4.

  Attributes:
    - generator (pl.LightningModule): The variable in which we save the value of the generator arg.
    - lr_generator (float): The variable in which we save the value of the lr_generator arg.
    - lr_discriminator (float): The variable in which we save the value of the lr_discriminator arg.
    - example_input_array (torch.Tensor): An example of the type of torch.Tensor input of the model.
  """
  def __init__(self,
               generator: pl.LightningModule,
               lr_generator: float = 1e-4,
               lr_discriminator: float = 1e-4):
    super().__init__()

    self.generator = generator
    self.discriminator = Discriminator()
    self.lr_generator = lr_generator
    self.lr_discriminator = lr_discriminator
    self.example_input_array = torch.rand(16, 3, 32, 32)
  
  def forward(self,
              x: torch.Tensor) -> torch.Tensor:
    """The forward pass.

    Args:
      - x (torch.Tensor): The input torch.Tensor.

    Returns:
      - torch.Tensor : the output of the generator.
    """
    x = self.generator(x)
    return x
  
  def generator_step(self,
                     x: torch.Tensor,
                     y: torch.Tensor) -> float:
    """ The generator step.

    Args:
      - x (torch.Tensor): The low resolution input image.
      - y (torch.Tensor): The high resolution output image.

    Returns:
      - gen_wasserstein_loss (float): The generator loss.
    """
    gen_imgs = self(x)

    discr_output = self.discriminator(gen_imgs)

    gen_wasserstein_loss = - torch.mean(discr_output)

    self.log("gen_wasserstein_loss", gen_wasserstein_loss, on_step=True, prog_bar=True)
    
    return gen_wasserstein_loss
  
  def generator_step2(self,
                     x: torch.Tensor,
                     y: torch.Tensor) -> float:
    """ The generator step.

    Args:
      - x (torch.Tensor): The low resolution input image.
      - y (torch.Tensor): The high resolution output image.

    Returns:
      - gen_loss (float): The generator loss.
    """
    gen_imgs = self(x)

    discr_output = self.discriminator(gen_imgs)
    h_discr_output = self.discriminator(y.detach().clone())

    l1_loss = 0.01*nn.L1Loss()(gen_imgs, y)
    self.log("l1_loss", l1_loss, on_step=True, prog_bar=True)

    h_adv_loss = torch.mul(nn.BCEWithLogitsLoss()(h_discr_output - torch.mean(discr_output), 
                                        torch.full([discr_output.size(0), 1], 0.0).cuda()), 0.5)
    
    s_adv_loss = torch.mul(nn.BCEWithLogitsLoss()(discr_output - torch.mean(h_discr_output),
                                       torch.full([discr_output.size(0), 1], 1.0).cuda()), 0.5)
    
    adv_loss = 0.005*(h_adv_loss + s_adv_loss)
    
    self.log("adv_loss", adv_loss, on_step=True, prog_bar=True)

    content_loss = content_criterion(gen_imgs, y)
    self.log("content_loss", content_loss, on_step=True, prog_bar=True)

    gen_loss = adv_loss + content_loss + l1_loss

    self.log("gen_loss", gen_loss, on_step=True, prog_bar=True)
    
    return gen_loss

  def discriminator_step(self,
                         fake: torch.Tensor,
                         real: torch.Tensor) -> float:
    """The discriminator step.

    Args:
      - fake (torch.Tensor): The fake image (The generated one).
      - real (torch.Tensor): The real image.

    Returns:
      - dis_wasserstein_loss (float): The discriminator loss.
    """

    h_discr_output = self.discriminator(real)
    gen_imgs = self(fake)
    discr_output = self.discriminator(gen_imgs.detach().clone())

    dis_wasserstein_loss = ( torch.mean(discr_output) - torch.mean(h_discr_output) )
    self.log("dis_wasserstein_loss", dis_wasserstein_loss, on_step=True, prog_bar=True)
    
    return dis_wasserstein_loss

  def discriminator_step2(self,
                         fake: torch.Tensor,
                         real: torch.Tensor) -> float:
    """The discriminator step.

    Args:
      - fake (torch.Tensor): The fake image (The generated one).
      - real (torch.Tensor): The real image.

    Returns:
      - dis_loss (float): The discriminator loss.
    """

    h_discr_output = self.discriminator(real)

    gen_imgs = self(fake)
    discr_output = self.discriminator(gen_imgs.detach().clone())

    loss_real = torch.mul(nn.BCEWithLogitsLoss()(h_discr_output - torch.mean(discr_output), 
                                        torch.full([discr_output.size(0), 1], 1.0).cuda()), 0.5)

    loss_fake = torch.mul(nn.BCEWithLogitsLoss()(discr_output - torch.mean(h_discr_output), 
                                        torch.full([discr_output.size(0), 1], 0.0).cuda()), 0.5)
    
    dis_loss = (loss_real + loss_fake)

    self.log("dis_loss", dis_loss, on_step=True, prog_bar=True)
    
    return dis_loss
  
  def training_step(self,
                    batch: Tuple[torch.Tensor, torch.Tensor],
                    batch_idx: int,
                    optimizer_idx: int) -> float:
    """A very simple training step.

    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch index.
      - optimizer_idx (int): The index of the optimizer, 0 if we're training the discriminator for the wasserstein_loss, 
                            1 if we're training the discriminator for the Total Loss used in the ESRGAN, 2 if we're training the generator for the wasserstein_loss, 
                            3 if we're training the generator for the Total Loss used in the ESRGAN.

    Returns:
      - loss (float): The training loss.
    """
    x, y = batch

    if optimizer_idx == 0:
      loss = self.discriminator_step(x,y)
      return loss
    elif optimizer_idx == 1:
      loss = self.discriminator_step2(x,y)
      return loss
    elif optimizer_idx == 2:
      loss = self.generator_step(x,y)
      return loss
    elif optimizer_idx == 3:
      loss = self.generator_step2(x,y)    
      return loss

    return None

  def test_step(self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                batch_idx: int) -> float:
    """A very simple test step.

    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch index.

    Returns:
      - psnr (float): The test loss defined as the mean of the PSNR on the test set.
    """
    x, y = batch
    y_hat = norm_zero_255(self(x).detach().clone())
    y = norm_zero_255(y.detach().clone())
    psnr = PSNR(y_hat, y)
    self.log('test_psnr', psnr, on_epoch=True, prog_bar=True)
    return psnr
  
  def validation_step(self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                batch_idx: int) -> float:
    """A very simple validation step.

    Args:
      - batch (Tuple[torch.Tensor, torch.Tensor]): The batch.
      - batch_idx (int): The batch index.

    Returns:
      - psnr (float): The validation loss defined as the mean of the PSNR on the validation set.
    """
    x, y = batch
    y_hat = norm_zero_255(self(x).detach().clone())
    y = norm_zero_255(y.detach().clone())
    psnr = PSNR(y_hat, y)
    self.log('val_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
    return psnr

  def optimizer_step(self,
                     epoch: int,
                     batch_idx: int,
                     optimizer: torch.optim.Optimizer,
                     optimizer_idx: int,
                     optimizer_closure: Optional[Callable[[], Any]],
                     on_tpu: bool,
                     using_native_amp: bool,
                     using_lbfgs: bool) -> None:
    """The optimizer step in which we clamp the weights before training of the generator.

    Args:
      - epoch (int): The current epoch.
      - batch_idx (int): The batch index.
      - optimizer (torch.optim.Optimizer): The optimizer.
      - optimizer_idx (int): The optimizer index.
      - optimizer_closure (Optional[Callable[[], Any]]): The optimizer closure.
      - on_tpu (bool): ``True`` if TPU backward is required.
      - using_native_amp (bool): ``True`` if using native amp.
      - using_lbfgs (bool): True if the matching optimizer is :class:`torch.optim.LBFGS`.
    """
    optimizer.step(closure=optimizer_closure)
    
    if optimizer_idx == 1:
      for p in self.discriminator.parameters():
        p.data.clamp_(-0.01,0.01)

  def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, int]]:
    """The configurization of the optimizers.

    Returns:
      - Dict[str, Union[torch.optim.Optimizer, int]] : The dictionary with the optimizers and thier frequency.
    """
    g_optimizer1 = torch.optim.RMSprop(self.generator.parameters(), lr=5e-5)
    d_optimizer1 = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
    g_optimizer2 = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
    d_optimizer2 = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
    return ( {'optimizer': d_optimizer1, 'frequency': 5},
             {'optimizer': d_optimizer2, 'frequency': 1},
             {'optimizer': g_optimizer1, 'frequency': 1}, 
             {'optimizer': g_optimizer2, 'frequency': 1} )

#Global Variables
feature_model_extractor_node = "features.34"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]
content_criterion = ContentLoss(feature_model_extractor_node, feature_model_normalize_mean, feature_model_normalize_std)
content_criterion = content_criterion.to(device=torch.device("cuda",0), memory_format=torch.channels_last)

if __name__ == "__main__":
    esrgan_generator = Generator.load_from_checkpoint('./weights/ESRGAN/ESRGAN-Generator/ESRGAN-Generator.ckpt',
                                     in_channels=3, out_channels=64, numBlocks=16, beta=0.2)

    esrgan_gan = GAN.load_from_checkpoint('./weights/ESRGAN/ESRGAN-GAN/ESRGAN-GAN.ckpt', 
                                       generator=Generator(in_channels=3, out_channels=64, numBlocks=16, beta=0.2) )


    