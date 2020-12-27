from glob import glob
import yaml

train_img_list = glob('data/images_car/train/images/*.jpg')
valid_img_list = glob('data/images_car/valid/images/*.jpg')
print(len(train_img_list), len(valid_img_list))



def createtxt():
  with open('data/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list))
  with open('data/valid.txt', 'w') as f:
    f.write('\n'.join(valid_img_list))


with open('data/data.yaml','r') as f:
  data=yaml.safe_load(f)
print(data)

data['train']='data/train.txt'
data['val']='data/valid.txt'

with open('data/data.yaml','w') as f:
  yaml.safe_dump(data,f)

print(data)
# python train.py --img 416 --batch 10 --epochs 100 --data /data/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name car_yolov5s
# val_img_path="/gdrive/MyDrive/drivemovie.mp4"
# !python detect.py --weights best.pt --img 416 --conf 0.5 --source "car_test.mp4"
# Image(os.path.join('content/dataset/valid/images'),os.path.basename(val_I have answered the mail, so please check it.img_path))