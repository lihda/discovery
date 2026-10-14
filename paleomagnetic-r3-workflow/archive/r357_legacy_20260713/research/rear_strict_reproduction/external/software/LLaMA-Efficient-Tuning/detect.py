import GPUtil
import time
import os
import subprocess

def main():
    availible_cnt = 2
    except_ids = set()
    while availible_cnt:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            # 检查每个 GPU 的空闲显存是否大于 78000MB
            if gpu.memoryFree > 78000 and gpu.id not in except_ids:
                print(f"GPU {gpu.id}")
                subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu.id} nohup python ../usere/gen-neg.py > log/s-{gpu.id}.log 2>&1 &", shell=True)
                except_ids.add(gpu.id)
                availible_cnt -= 1
                if availible_cnt == 0:
                    break
        time.sleep(10)

if __name__ == "__main__":
    main()

