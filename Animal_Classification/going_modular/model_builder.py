"""
Contains functions to create PyTorch models such as.

1.  `create_ResNet50` - used to create ResNet_50 model
2.  `create_ResNet18` - used to create ResNet_18 model
3.  `create_wide_resnet101_2` - Used to create wide_resnet_102_2 model
4.  `create_AlexNet`  - Used to create AlexNet model
5.  `create_efficientnet_b0` - used to create EfficientNet_b0 model
6.  `create_effnetb2` - Used to create EfficientNet_b2 model
7.  `create_EfficientNet_V2`  - Used to EfficientNet_V2 model
8.  `create_MobileNet` - Used to create MobileNet model
"""
import torchvision
from typing import Tuple, Dict, List
from torch import nn
from torchvision import datasets, transforms
import torch
def create_ResNet50(number_of_classes : int = 2,
                    trainable : bool = False,
                    trainable_layers = 0,
                    device: torch.device = 'cpu'):
  """Building ResNet50_Transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    trainable_layers: Int repersents the number of trainable layers you want
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    ResNet50_model, ResNet50_transforms = create_ResNet50(number_of_classes = 12,
                                                      trainable = True,
                                                      trainable_layers = 0,
                                                      device = 'cuda')
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.ResNet50_Weights.DEFAULT
  model = torchvision.models.resnet50(weights =Weights ).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  ResNet50_model_Layers = [model.layer1,model.layer2,model.layer3,model.layer4]
  if(trainable == True):
    for i in range((trainable_layers),4):
      for param in ResNet50_model_Layers[i].parameters():
        param.requires_grad = False
  else:
    for layer in ResNet50_model_Layers:
      for param in layer.parameters():
        param.requires_grad = False
  # 4. Change the classifier head
  in_fearues = model.fc.in_features
  model.fc = nn.Linear(in_features = in_fearues, out_features = number_of_classes, bias=True)
  # 5. Give the model a name
  model.name = "ResNet_50"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.fc}")

  #Return the model and the transform
  return model, transform
"""___________________________________________________________________________________________"""

def create_ResNet18(number_of_classes : int = 2,
                    trainable : bool = False,
                    trainable_layers = 0,
                    device: torch.device = 'cpu'):
  """Building ResNet18_Transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    trainable_layers: Int repersents the number of trainable layers you want
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    ResNet18_model, ResNet18_transforms = create_ResNet18(number_of_classes = 12,
                                                      trainable = True,
                                                      trainable_layers = 0,
                                                      device = 'cuda')
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.ResNet18_Weights.DEFAULT
  model = torchvision.models.resnet18(weights =Weights ).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  ResNet18_model_Layers = [model.layer1,model.layer2,model.layer3,model.layer4]
  if(trainable == True):
    for i in range((trainable_layers),4):
      for param in ResNet18_model_Layers[i].parameters():
        param.requires_grad = False
  else:
    for layer in ResNet18_model_Layers:
      for param in layer.parameters():
        param.requires_grad = False
  # 4. Change the classifier head
  in_fearues = model.fc.in_features
  model.fc = nn.Linear(in_features = in_fearues, out_features = number_of_classes, bias=True)
  # 5. Give the model a name
  model.name = "ResNet_18"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.fc}")

  #Return the model and the transform
  return model, transform

"""___________________________________________________________________________________________"""
def create_wide_resnet101_2(number_of_classes : int = 2,
                            trainable : bool = False,
                            trainable_layers = 0,
                            device: torch.device = 'cpu'):
  """Building Wide_ResNet101_2_Transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    trainable_layers: Int repersents the number of trainable layers you want
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    Wide_ResNet102_2_model, Wide_ResNet102_2_tramsform = create_wide_resnet101_2(number_of_classes = 12,
                                                                                trainable = True,
                                                                                trainable_layers = 0,
                                                                                device = 'cuda')
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT
  model = torchvision.models.wide_resnet101_2(weights=Weights).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  wide_resnet_101_2_model_Layers = [model.layer1,model.layer2,model.layer3,model.layer4]
  if(trainable == True):
    for i in range((trainable_layers),4):
      for param in ResNet50_model_Layers[i].parameters():
        param.requires_grad = False
  else:
    for layer in ResNet50_model_Layers:
      for param in layer.parameters():
        param.requires_grad = False
  # 4. Change the classifier head
  model.fc = nn.Linear(in_features=2048, out_features=number_of_classes, bias=True)
  # 5. Give the model a name
  model.name = "Wide_resnet_102_2"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.fc}")

  #Return the model and the transform
  return model, transform
"""___________________________________________________________________________________________"""

def create_AlexNet (number_of_classes : int = 2,
                    trainable : bool = False,
                    device: torch.device = 'cpu'):
  """Building AlexNet_Transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    AlexNet_model, AlexNet_transforms = create_AlexNet(number_of_classes = 12,
                                                      trainable = True)
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.AlexNet_Weights.DEFAULT
  model = torchvision.models.alexnet(weights =Weights ).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  if(trainable == False):
    for param in model.features.parameters():
      param.requires_grad = False
  # 4. Change the classifier head
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=number_of_classes, bias=True)
  )
  # 5. Give the model a name
  model.name = "AlexNet"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.classifier}")

  #Return the model and the transform
  return model, transform


"""___________________________________________________________________________________________"""

def create_efficientnet_b0 (number_of_classes : int = 2,
                    trainable : bool = False,
                    device: torch.device = 'cpu'):
  """Building EfficientNet_B0 transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    EfficientNet_B0_model, EfficientNet_B0_transforms = create_efficientnet_b0(number_of_classes = 12,
                                                                              trainable = False)
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  model = torchvision.models.efficientnet_b0(weights=Weights).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  if(trainable == False):
    for param in model.features.parameters():
      param.requires_grad = False
  # 4. Change the classifier head
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=number_of_classes, bias=True)
  )

  # 5. Give the model a name
  model.name = "EfficientNet_B0"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.classifier}")

  #Return the model and the transform
  return model, transform
"""___________________________________________________________________________________________"""
def create_MobileNet(number_of_classes : int = 2,
                    trainable : bool = False,
                    device: torch.device = 'cpu'):
  """Building MobileNet transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    MobileNet_model, MobileNet_transform = create_MobileNet(number_of_classes = 12,
                                                            trainable = False)
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.DEFAULT
  model = torchvision.models.mobilenet_v2(weights=Weights).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  if(trainable == False):
    for param in model.features.parameters():
      param.requires_grad = False
  # 4. Change the classifier head
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=1280, out_features=number_of_classes, bias=True)
  )

  # 5. Give the model a name
  model.name = "MobileNet"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.classifier}")
  #Return the model and the transform
  return model, transform

"""___________________________________________________________________________________________"""
def create_EfficientNet_V2(number_of_classes : int = 2,
                          trainable : bool = False,
                          device: torch.device = 'cpu'):
  """Building EfficientNet_V2 transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    EfficientNet_V2_model, EfficientNet_V2_transform = create_EfficientNet_V2(number_of_classes = 12,
                                                                              trainable = False)
  """

  # 1. Get the base model with pretrained weights and send to target device
  Weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
  model = torchvision.models.efficientnet_v2_l(weights = Weights).to(device)
  # 2.  Get The suitable transformation of the model
  transform = Weights.transforms()
  # 3.  Freeze the base model layers
  if(trainable == False):
    for param in model.features.parameters():
      param.requires_grad = False
  else:
    pass
  # 4. Change the classifier head
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features=1280, out_features=number_of_classes, bias=True)
)

  # 5. Give the model a name
  model.name = "EfficientNet_V2"
  print(f"[INFO] Created new {model.name} model.")
  print(f"[INFO] With classifier layer {model.classifier}")

  #Return the model and the transform
  return model, transform
"""___________________________________________________________________________________________"""

def create_effnetb2(number_of_classes : int = 2,
                    trainable : bool = False,
                    device: torch.device = 'cpu'):
  """Building EffnetB0 transfer Model.

  Args:
    number_of_classes : Int represents the number of classes of the model to adjust the last layer.
    trainable: Bool represents if you want the parameters of the model to be trainable or not.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Return:
    Return the implemented model and it's suitable transform
  Example usage:
    EfficientNet_B2_model, EfficientNet_B2_transform = create_effnetb2(number_of_classes = 12,
                                                                              trainable = False)
  """

  # 1. Get the base model with pretrained weights and send to target device
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  model = torchvision.models.efficientnet_b2(weights=weights).to(device)

  # 2. Freeze the base model layers
  if(trainable):
    for param in model.features.parameters():
        param.requires_grad = False

  # 3. Set the seeds
  set_seeds()

  # 4. Change the classifier head
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=1408, out_features=number_of_classes)
  ).to(device)

  # 5. Give the model a name
  model.name = "effnetb2"
  print(f"[INFO] Created new {model.name} model.")
  return model
