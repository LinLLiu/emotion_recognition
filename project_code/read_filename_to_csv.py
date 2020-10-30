import pandas as pd
import os

save_name = r"E:\emotion_recognition\audios\AUDIO.csv"
dir_path = r"E:\emotion_recognition\audios\cutresults"


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return (dir_list)


wav_name = get_file_list(dir_path)
wav_name = pd.DataFrame(wav_name,columns=['name'])
wav_name.to_csv(save_name, encoding="utf_8", index=False)
