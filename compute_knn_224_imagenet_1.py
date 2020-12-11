import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy import misc
from scipy.stats import truncnorm
from pytorch_pretrained_biggan import save_as_images
import save_images

aug_dir = './image331.png'

IMAGENET_PATH = '/data/nzl17/ImageNet/raw-data/imagenet-data/train' # '/home/imagenet/train/'
class_id = 'n02106166/'

paths = glob.glob(os.path.join(IMAGENET_PATH, class_id, '*.JPEG'))
aug_img = Image.open(aug_dir)

trans1 = transforms.Compose([
    transforms.ToTensor()
])

trans2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# AUG = trans1(aug_img)
AUG = trans1(aug_img)
d = {}
mse_loss = torch.nn.MSELoss(reduction='sum')
index = 0

for path in paths:
    img = Image.open(path)
    # print(path)
    IMG = trans2(img)
    IMG = IMG.view(1, IMG.size(0), IMG.size(1), IMG.size(2))
    IMG = torch.nn.functional.interpolate(IMG, (512, 512), mode='bilinear', align_corners=True)
    img_id = path.split('/')[-1]

    loss = mse_loss(IMG, AUG)
    d[str(img_id)] = loss.data.item()
    print('index:{0} img_id:{1} loss:{2}'.format(index, img_id, loss.data.item()))
    index += 1

print('len(d):', len(d))
print('aug_dir:', aug_dir)
sorted_d = sorted(d.items(), key=lambda item: item[1])
result = sorted_d[:5]
print(result)

basic_path = os.path.join('./', aug_dir.split('/')[1].split('.')[0])
isExists = os.path.exists(basic_path)
if not isExists:
    os.makedirs(basic_path)
print('basic_path:', basic_path)

id = 1
for r in result:
    raw_img_id = r[0]
    raw_img_dir = os.path.join(IMAGENET_PATH, class_id, raw_img_id)
    img = Image.open(raw_img_dir)
    img_raw = trans1(img)
    img_raw = img_raw.view(1, img_raw.size(0), img_raw.size(1), img_raw.size(2))
    img_224 = trans2(img)
    img_224 = img_224.view(1, 3, 224, 224)
    img_512 = torch.nn.functional.interpolate(img_224, (512, 512), mode='bilinear', align_corners=True)

    save_images.save_images(img_raw.cpu().numpy(), '{0}/{1}_{2}_raw'.format(basic_path, id, raw_img_id.split('.')[0]))
    save_images.save_images(img_512.cpu().numpy(), '{0}/{1}_{2}_512'.format(basic_path, id, raw_img_id.split('.')[0]))

    id += 1