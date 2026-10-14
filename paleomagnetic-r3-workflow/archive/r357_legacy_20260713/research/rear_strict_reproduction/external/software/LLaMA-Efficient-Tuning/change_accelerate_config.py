import sys
import yaml

def update_config(num_gpus, main_port):
    # 读取config.yaml文件
    with open('config', 'r') as file:
        config = yaml.safe_load(file)

    # 修改num_processes的值
    config['num_processes'] = num_gpus
    config['main_process_port'] = main_port

    # 写回config.yaml文件
    with open('config', 'w') as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    num_gpus = len(sys.argv[1].split(','))
    main_port = int(sys.argv[2])
    update_config(num_gpus, main_port)
