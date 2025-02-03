"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""
import os

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import shutil
import random
from matplotlib import pyplot as plt
NUM_WORKERS = os.cpu_count()
import torchvision

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    val_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose = None,
    test_transform: transforms.Compose = None,
    batch_size: int=32,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    val_dir: Path to validaiting directory.
    train_transform: torchvision transforms to perform on training data.
    val_transform: torchvision transforms to perform on validating data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader,validate_dataloader ,test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             val_dir = path/to/val_dir,
                             test_dir=path/to/test_dir,
                             train_transform=some_transform,
                             val_transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=train_transform)
  val_data = datasets.ImageFolder(test_dir, transform=val_transform)
  test_data = datasets.ImageFolder(test_dir, transform=test_transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  val_dataloader = DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )
  return train_dataloader,val_dataloader, test_dataloader, class_names
"""--------------------------------------------------------------------------"""
def image_from_Excel_to_folder(source_path : str,
                              destination_path :str,
                              excel_file : pd,
                              number_of_images : int=20000,
                              train_size : float=0.8,
                              val_size : float=0.1,
                              test_size : float=0.1,
                              class_names : List[str]=["smile","no_smile"],
                              ):
  """ This function takes the source_path of the images, excel file and some args and make your data ready for PyTorch to use
  , it converts them into
  Data/
    Train_Data/
      class_1/....
      class_2/
      ...
    Test_Data/
      class_1/...
      class_2/...
      ...
    Val_Data/
      class_1/...
      class_2/...
      ...
  Args:
    source_path : The source path of the images in str format
    destination_path : The desitination of the images to be stored in str format
    excel_file : The excel file of the data in format [image_name, class_id] in pd format
    number_of_images : Number of images you want to move to your desitnation folder
    train_size : The train size of your data in perecentage
    val_size : The val size of your data in  perecentage
    test_size : The test size of your data in perecentage
    class_names : The class names of your data so that it can separate them into folders
  Example:
    ImageFolder(source_path = "data/data_faces/img_align_celeba",
                destination_path = "data",
                excel_file = df,
                number_of_images = 20000,
                train_size = 0.8,
                val_size = 0.1,
                test_size = 0.1,
                class_names = ["smile","no_smile"],
                )
  """
  source_path = Path(source_path)
  destination_path = Path(destination_path)
  train_data_path = destination_path/"train_data"
  validate_data_path = destination_path/"validate_data"
  test_data_path = destination_path/"test_data"
  df = excel_file
  # Check if the folders are created and if not make ones
  Pathes = [train_data_path,validate_data_path,test_data_path]
  for i in Pathes:
    if(i.is_dir()):
      print(f"The Directory {i} is already exists, skipping ...")
    else:
      print(f"The Directory {i} is not exists, making one... ")
      i.mkdir(parents=True, exist_ok=True)
    for j in class_names:
      class_path = i/j
      if(class_path.is_dir()):
        print(f"  The Directory {class_path} is already exists, skipping ...")
      else:
        print(f"  The Directory {class_path} is not exists, making one... ")
        class_path.mkdir(parents=True, exist_ok=True)

  # Get image by image and see its class and out it in the desired folder
  train_size = int(0.8*number_of_images)
  val_size =   int(0.1*number_of_images)
  test_size =  number_of_images - train_size-val_size
  image_counter = 0
  train_smile, train_no_smile = 0, 0
  val_smile, val_no_smile = 0, 0
  test_smile, test_no_smile = 0, 0
  for i, (_, image) in enumerate(df.iterrows()):
    source_image = source_path/image['image_id']
    if(source_image.exists()):
      image_counter+=1
      # Train Images
      if(image['Smiling']==1 and train_smile< train_size/2):
        destination_index = 0
        class_index =0
        train_smile+=1
      elif(image['Smiling']==-1 and train_no_smile< train_size/2):
        destination_index = 0
        class_index =1
        train_no_smile+=1
      # Val Images
      elif(image['Smiling']==1 and val_smile< val_size/2):
        destination_index = 1
        class_index =0
        val_smile+=1
      elif(image['Smiling']==-1 and val_no_smile< val_size/2):
        destination_index = 1
        class_index =1
        val_no_smile+=1
      # Test Imges
      elif(image['Smiling']==1 and test_smile< test_size/2):
        destination_index = 2
        class_index =0
        test_smile+=1
      elif(image['Smiling']==-1 and test_no_smile< test_size/2):
        destination_index = 2
        class_index =1
        test_no_smile+=1
      Destination_image_path = Pathes[destination_index]/class_names[class_index]
      if(image_counter%1000==0):
        print(f"Finnished {(image_counter/number_of_images)*100} % of the data")
      try:
        shutil.copyfile(source_path/image['image_id'], Destination_image_path/image['image_id'])
      except:
        print(f"The Directory was not found!")
        continue
    else:
      print(f"Image {source_image} was not found!")
    if(image_counter==number_of_images):
      print("Finnished Copying")
      print(f"train smile {train_smile} | train no smile {train_no_smile}")
      print(f"Val smile   {val_smile} | Val no smile {val_no_smile}")
      print(f"test smile {test_smile} | test no smile {test_no_smile}")
      return


"""-----------------------------------------------------------------------------------------------------"""
# 1.  Create a function to take in a dataset
def display_random_images(data_dir : str,
                          transforms : torchvision.transforms = None,
                          fig_size = (20,10),
                          n : int = 10,
                          display_shape : bool = True,
                          seed : int = None,
                          permute : bool = False
                          ):
  """Plot random images to visualize the data.

  Args:
      data_dir: the directory of the data you want to visualize
      fig_size : represent the size of the figure you want
      classes (List(str)) : The classes of the target , Default = None
      n (int) : The number of random of images you want to display, Default = 10
      display_shape (bool) : Boolean to determine the shape of the display
      seed (int) : Determines the seed random number
      Permute(Bool) : To display image the image must be in format (H,W,C) if the image is in another format them you need permute = True

  Returns:
      Nothing, Plot random number of images from the given dataset.
  """

  # 2.  Adjust the display if n is too high
  dataset = datasets.ImageFolder(data_dir,transform=transforms)
  classes = dataset.classes

  if n > 10:
    n = 10
    display_shape = False
    print(f"For dsplay, n shouldn't be larger than 10, setting to 10 and removing shape display")

  # 3.  Set the random seed
  if seed:
    random.seed(seed)

  # 4.  Get random sample indexes
  random_samples_idx = random.sample(range(len(dataset)), k=n)

  # 5.  Setup the plot
  plt.figure(figsize=fig_size)

  # 6.  Loop through random indexes and plot them with matplolib
  for i , targ_sample in enumerate(random_samples_idx):
    targ_image, targ_label = dataset[targ_sample][0],dataset[targ_sample][1]

    #7. Adjust tensor dimension for plotting
    if(permute):
      targ_image = targ_image.permute(1,2,0)   #[color_channel, height, width]   ->    [height, width, color_channel]
    plt.subplot(1,n,i+1)
    plt.imshow(targ_image)
    if classes:
        title=f"Class: {classes[targ_label]}"
        if display_shape:
          height = targ_image.height
          width = targ_image.width
          shape = [height,width]
          title+=f"\nShape:{shape}"
    plt.title(title)
    plt.axis(False)

