# Pix2Pix_tf2.0

## 設定環境及安裝套件


1. 下載 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ([安裝教學](https://www.1989wolfe.com/2019/07/miniCONDAwithPython.html))

2. 開啟命令提示字元並創建虛擬環境:

    ```
    conda create -n tf2 python=3.7
    ```

3. 進入虛擬環境

    ```
    conda activate tf2
    ```
   
4. 使用 Conda 安裝 CUDA 相關套件
    ```
    conda install cudatoolkit=10.1 cudnn
    ```
5. 使用 Pip 安裝必要套件
    ```
    pip install -r requirements.txt
    ```

## Usage

1. 進行訓練
    ```
    python train.py
    ```
2. 查看 TensorBoard
    ```
    tensorboard --logdir results --port 8008
    ```   
   透過瀏覽器輸入
    ```
    http://localhost:8008/
    ```
   
3. 訓練完成後，進行 Inference
    ```
    python test.py --load_model_path <YOUR_MODEL_PATH>
    ```   
