import fire
import os
import pandas as pd

def split(input_path, output_path, cuda_list):
    
    if isinstance(cuda_list, int):
        cuda_list = [str(cuda_list)]
    elif isinstance(cuda_list, str):
        if ',' in cuda_list:
            cuda_list = cuda_list.split(',')
        else:
            cuda_list = cuda_list.split()
    elif isinstance(cuda_list, tuple):
        cuda_list = list(cuda_list)
    
    # 去除可能的空格
    cuda_list = [str(x).strip() for x in cuda_list if str(x).strip()]

    # 读取CSV文件
    df = pd.read_csv(input_path)
    
    df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_list = list(cuda_list)
    cuda_num = len(cuda_list)
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        df[start:end].to_csv(f'{output_path}/{cuda_list[i]}.csv', index=True)
        
if __name__ == '__main__':
    fire.Fire(split)
