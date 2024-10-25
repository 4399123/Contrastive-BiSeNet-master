#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

#·������
onnx_path=r'./onnx/best-smi.onnx'
pic_path=r'./onnx/11.jpg'
w,h=512,512

#��ɫ������
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0]=[255,255,255]
palette[1]=[0,255,0]
palette[2]=[0,0,255]
palette[3]=[255,0,0]
palette[4]=[255,255,0]
palette[5]=[255,0,255]
palette[6]=[171,130,255]
palette[7]=[155,211,255]
palette[8]=[0,255,255]

mean=(120,114,104)
std=(70,69,73)

#onnxģ������
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])


#����ͼ��Ԥ����
img=Image.open(pic_path).resize((w,h))
imgbak=img.copy()
img=np.array(img).astype(np.float32)  # ע������typeһ��Ҫnp.float32
img-=mean                             #����ֵ
img/=std                              #������
img=np.array([np.transpose(img,(2,0,1))])

#ģ������
out = session.run(None,input_feed = { 'input' : img })
# out=np.argmax(out[0],axis=1)
out=out[0].astype('int')
pred= palette[out].squeeze()


#����ͼ��
n=0
cv2.imwrite('./onnx/mask_{}.jpg'.format(n), pred)

img1=np.array(imgbak)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img=cv2.addWeighted(img1,0.3,pred,0.7,0)
cv2.imwrite('./onnx/out_{}.jpg'.format(n), img)


