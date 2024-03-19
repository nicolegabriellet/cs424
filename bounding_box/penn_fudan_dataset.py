import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "nicole_oo_images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "nicole_oo_labels"))))

    def __getitem__(self, idx):
      # load images and masks
      img_path = os.path.join(self.root, "nicole_oo_images", self.imgs[idx])
      mask_path = os.path.join(self.root, "nicole_oo_labels", self.masks[idx])
      img = read_image(img_path).float() / 255.0
      mask = read_image(mask_path)
      # instances are encoded as different colors
      obj_ids = torch.unique(mask)
      # first id is the background, so remove it
      obj_ids = obj_ids[1:]
      num_objs = len(obj_ids)
      
      # split the color-encoded mask into a set
      # of binary masks (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
      masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
      # get bounding box coordinates for each mask
      boxes = masks_to_boxes(masks)
      
      # validate boxes
      valid_boxes = []
      # only add valid boxes
      for box in boxes:
          if (box[0] != box[2]) and (box[1] != box[3]):
              valid_boxes.append(box)
      boxes = torch.stack(valid_boxes)
      
      # there is only one class
      labels = torch.ones((num_objs,), dtype=torch.int64)
      
      image_id = idx
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
      # suppose all instances are not crowd
      iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      
      # Wrap sample and targets into torchvision tv_tensors:
      img = tv_tensors.Image(img)
      
      # return img, target (as a dictionary with keys corresponding to those expected by your model)
      target = {}
      target['boxes'] = boxes
      target['labels'] = labels
      target['image_id'] = torch.tensor([image_id])
      target['area'] = area
      target['iscrowd'] = iscrowd
      target['masks'] = masks
      
      return img, target


    def __len__(self):
        return len(self.imgs)
  