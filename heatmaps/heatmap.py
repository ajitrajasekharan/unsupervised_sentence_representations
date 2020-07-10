import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def gen_heatmap(arr,output_file):
    sns.set()
    sns.set(font_scale=3)
    plt.figure(figsize=(50,40))
    ax = sns.heatmap(arr, vmin=0, vmax=1)
    plt.savefig(output_file)



if __name__ == "__main__":
    if (len(sys.argv) == 3):
        with open(sys.argv[1],'rb') as fp:
            arr = np.load(fp)
            gen_heatmap(arr,sys.argv[2])
    else:
        test_file = "/tmp/hm.png"
        np.random.seed(0)
        print("Usage: <input arr as numpy file> , <output file name>")
        print("Generarting a default arr and storing in:" + test_file)
        uniform_data = np.random.rand(42,42)
        gen_heatmap(uniform_data,test_file)

