import torch
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    # path = './isaacgymenvs/saved_models/flip/231124/'
    # name = "flip"
    # name = 'pos'
    # path = "isaacgymenvs/saved_models/pos/231218/"
    # name = 'pos_240230'
    # log_dir = "isaacgymenvs/runs/" + name +"/nn/"
    # name = 'flip_240129'
    # log_dir = "isaacgymenvs/saved_models/flip/zikang/"
    # name = 'angvel_diff'
    # name = 'pos_240230'
    # log_dir = "isaacgymenvs/saved_models/pos/240230/"
    # name = 'angvel_240312'
    name = 'actor_0'
    # name = 'agent'
    # log_time = '04-22-23-05'
    # log_dir = "isaacgymenvs/runs/QuadcopterVpp_pos/"
    # log_dir = "isaacgymenvs/runs/a_group_0/QuadcopterVpp_ppo_pos_240409/"

    # log_time = '04-26-10-00'
    # log_dir = "isaacgymenvs/runs/QuadcopterVpp_rotating/"
    #
    # log_time = '04-28-18-11'
    # log_time = '05-07-18-05'
    # log_time = '05-28-01-13'
    log_time = 'zhikun'
    log_dir = "isaacgymenvs/runs/gun_shoot/"

    input_size_list = [[1, 27]]
    # input_size_list = [[1, 5, 27]]  # 根据模型修改

    load_model_name = name + '.pt'
    save_model_name = name + '.rknn'
    load_model_path = log_dir + log_time + '/nn/' + load_model_name
    save_model_path = log_dir + log_time + '/nn/' + save_model_name


    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3566', 
                optimization_level=0, 
                float_dtype='float16',
                quantized_dtype = 'asymmetric_quantized-8')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=load_model_path,
                            input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(save_model_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(perf_debug=True)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    input_test_data = np.zeros(tuple(input_size_list[0]), dtype=np.float16)
    # input_test_data[...,6] = 1
    # input_test_data[...,16] = 1
    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_test_data])
    print("outputs", outputs)
    input_test_data = np.ones(tuple(input_size_list[0]), dtype=np.float16)
    # input_test_data[...,6] = 1
    # input_test_data[...,16] = 1
    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_test_data])
    print("outputs", outputs)
    print('done')

    # rknn.eval_perf()
    # rknn.accuracy_analysis(target='rk3566')
    
    rknn.release()
