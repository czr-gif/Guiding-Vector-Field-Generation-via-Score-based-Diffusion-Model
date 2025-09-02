from rknn.api import RKNN
import onnx
import numpy as np
import os

def export(taskname: str, modelname: str):
    """
    Export ONNX model to RKNN format for specific task and model type.

    Args:
        taskname (str): Name of the task, e.g. 'doublecircles' or 'SeperatedCircles'.
        modelname (str): Name of the model, e.g. 'score' or 'tangent'.
    """

    onnx_path = f"trans_model/{taskname}/{modelname}.onnx"
    rknn_path = f"trans_model/{taskname}/{modelname}.rknn"

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    rknn = RKNN()

    # 配置 RKNN 参数
    rknn.config(
        target_platform="rk3566",
        optimization_level=0,
        float_dtype="float16",
        quantized_dtype="asymmetric_quantized-8"
    )

    # 根据模型类型设置输入信息
    if modelname == "score":
        input_names = ["x", "t"]
        input_size_list = [[1, 2], [1, 1]]  # x_dim=2, t_dim=1
        print(f"[INFO] Loading ONNX score model from {onnx_path} (inputs: x:{2}, t:{1})")
    elif modelname == "tangent":
        input_names = ["input"]
        input_size_list = [[1, 2]]  # tangent_dim=2
        print(f"[INFO] Loading ONNX tangent model from {onnx_path} (input dim = 2)")
    else:
        raise ValueError(f"Unknown modelname: {modelname}. Expected 'score' or 'tangent'")

    # 加载 ONNX 模型
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=input_names,
        input_size_list=input_size_list,
        outputs=["output"]
    )
    if ret != 0:
        raise RuntimeError(f"Failed to load ONNX model: {onnx_path}")

    # 构建 RKNN 模型
    print("[INFO] Building RKNN model...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError("Failed to build RKNN model")

    # 导出 RKNN 模型
    print(f"[INFO] Exporting RKNN model to {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to export RKNN model to {rknn_path}")

    print(f"[SUCCESS] Model exported: {rknn_path}")




if __name__=='__main__':
    tasknames = ["experiment2"]
    modelnames = ["score", "tangent"]

    for task in tasknames:
        for model in modelnames:
            print(export(task, model))
    
