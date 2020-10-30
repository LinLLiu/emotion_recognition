from PIL import Image
import os
path = r"E:\emotion_recognition\pictures\neutral"
files = os.listdir(path)
files = [path + "\\" + f for f in files if f.endswith('.jpg')]
j = 0
for file in files:
    img = Image.open(file)
    new_img = img.resize((100,100), Image.BILINEAR)
    new_img.save(os.path.join(r"E:\emotion_recognition\pictures\neutral\face" + str(j) + '.jpg'))
    j = j + 1