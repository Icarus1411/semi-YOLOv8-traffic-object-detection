# Project5: Traffic Object Detection

本项目基于 AI3603-01-人工智能理论及应用 课程大作业Project5交通目标检测，项目成员为黄宇琪、李雪荟、杨凯。我们实现了YOLOv8全监督方法、YOLOv8伪标签方法和Paddle方法。

**项目总体目标：**使用给定的数据集进行训练，从而完成道路环境下十个类别的交通目标检测任务。类别包括：pedestrian、rider、car、bus、truck、train、bicycle, motorcycle、trafficlight和trafficsign。

文件及报告下载链接：https://jbox.sjtu.edu.cn/l/V1btuX   

## Yolov8

### 1. 创建环境

在 `.\ultralytics` 目录下运行以下指令（采用清华源进行配置）：

```bash
conda create -n yolo python=3.8.13
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或者根据 `requirements.txt` 安装以下库：

```bash
matplotlib>=3.3.0
numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0
```

### 2. 运行指令

可参考 `run.sh` 内分别对于训练、测试、验证、恢复训练的代码有一定模板：

```bash
# 从YAML构建一个新模型，从头开始训练
yolo detect train data= <.yaml文件路径> \
model= <.pt或.yaml文件路径> \
device=0 epochs=1000 imgsz=1280 seed=123 cos_lr=True lr0=0.02 lrf=1e-4 cache=True resume=True batch=16

# 伪标签训练
yolo detect train data= <.yaml文件路径> \
model= <.pt或.yaml文件路径> \
device=0 epochs=10 imgsz=1280 seed=123 cos_lr=True lr0=0.01 lrf=0.01 cache=True batch=32

#validation
yolo detect val data= <.yaml文件路径> model= <.pt或.yaml文件路径>

# 恢复训练
yolo train resume model= <.pt文件路径>
```

注意，代码运行前需对所使用的 `.yaml` 的数据路径进行修改，可参考 `./dataset/mydata.yaml`：

```yaml
# mydata
train: /lustre/home/acct-stu/stu299/AI-YOLO1/dataset0/yolov8_dataset/train  # train目录路径
val: /lustre/home/acct-stu/stu299/AI-YOLO1/dataset0/yolov8_dataset/valid # val 目录路径
test: /lustre/home/acct-stu/stu299/AI-YOLO1/dataset0/yolov8_dataset/test # test 目录路径

# Classes
names:
  0: pedestrian
  1: rider
  2: car
  3: bus
  4: truck
  5: train
  6: bicycle
  7: motorcycle
  8: trafficlight
  9: trafficsign
```

### 3. 代码说明

- `dataSplit.py` ：用于将数据集按指定比例划分为 train/valid/test；
- `json2txt.py` ：用于将 `.json` 文件转化为 `.txt` 文件；
- `myTrain.py`：用于通过预训练模型生成伪标签；
- `pred_out.py`：用于基于最终模型生成预测结果及其标准格式的 `.json` 文件



## PaddleDetection

半监督目标检测(Semi DET)是**同时使用有标注数据和无标注数据**进行训练的目标检测，既可以极大地节省标注成本，也可以充分利用无标注数据进一步提高检测精度。PaddleDetection团队提供了[DenseTeacher](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/semi_det/denseteacher)和[ARSL](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/semi_det/arsl)等最前沿的半监督检测算法。

在本次交通目标检测项目中，为了更好地利用无标注数据集，我们基于Paddle的半监督模型进行了训练与测试。

### 1.安装

#### 1.1 创建虚拟环境

##### 1.1.1 安装环境

首先根据具体的 Python 版本创建 Anaconda 虚拟环境，PaddlePaddle 的 Anaconda 安装支持 3.8 - 3.12 版本的 Python 安装环境。

```
conda create -n paddle_env python=YOUR_PY_VER
```

##### 1.1.2 进入 Anaconda 虚拟环境

```
conda activate paddle_env
```

#### 1.2 根据版本进行安装

##### 1.2.1 CPU 版的 PaddlePaddle

```
conda install paddlepaddle==2.6.0 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

##### 1.2.2 GPU 版的 PaddlePaddle[¶](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/conda/linux-conda.html#gpu-paddlepaddle)

- 对于 `CUDA 11.2`，需要搭配 cuDNN 8.2.1(多卡环境下 NCCL>=2.7)，安装命令为:

```
conda install paddlepaddle-gpu==2.6.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

- 对于 `CUDA 11.6`，需要搭配 cuDNN 8.4.0(多卡环境下 NCCL>=2.7)，安装命令为:

```
conda install paddlepaddle-gpu==2.6.0 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

- 对于 `CUDA 11.7`，需要搭配 cuDNN 8.4.1(多卡环境下 NCCL>=2.7)，安装命令为:

```
conda install paddlepaddle-gpu==2.6.0 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

#### 1.3 验证安装

安装完成后可以使用 `python3` 进入 python 解释器，输入`import paddle` ，再输入 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明已成功安装。



### 2. 半监督数据集准备

半监督数据集处理的代码架构：

```bash
/proc_code
	- labels_trainval.json # 本项目中给定的标注json文件
	- proc_lb.py # 生成有标签（label）的数据集
	- proc_ulb.py # 生成无标签（unlabel）的数据集
	- proc_train_eval.py # 将有标签数据集按一定比例分割为train集和eval集
	- proc_train_eval_test.py # 将有标签数据集按一定比例分割为train、eval和test集
```



### 3. 修改配置文件

- **configs/semi_det/\_base\_/coco\_detection.yml**

  ```yaml
  metric: COCO
  num_classes: 10 # 修改类别数目 
  
  # partial labeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
  TrainDataset:
    !SemiCOCODataSet
      image_dir: images/trainval # 修改有标签图片路径
      anno_path: coco_lb_train.json # 修改train标注路径
      dataset_dir: /lustre/home/acct-stu/stu050/dataset # 修改数据集路径
      data_fields: ['image', 'gt_bbox', 'gt_class']
      load_crowd: true
  
  # partial unlabeled COCO, use `SemiCOCODataSet` rather than `COCODataSet`
  UnsupTrainDataset:
    !SemiCOCODataSet
      image_dir: unlabel_images # 修改有标签图片路径
      anno_path: coco_ulb.json # 修改无标注json文件路径
      dataset_dir: /lustre/home/acct-stu/stu050/dataset # 修改数据集路径
      data_fields: ['image']
      supervised: False
  
  EvalDataset:
    !COCODataSet
      image_dir: images/trainval # 修改有标签图片路径
      anno_path: coco_lb_eval.json # 修改eval标注路径
      dataset_dir: /lustre/home/acct-stu/stu050/dataset # 修改数据集路径
      allow_empty: true
  
  TestDataset:
    !ImageFolder
      anno_path: coco_lb_test.json # 修改test标注路径
      dataset_dir: /lustre/home/acct-stu/stu050/dataset # 修改数据集路径
  ```

- **configs/semi_det/arsl/\_base\_/optimizer_90k.yml**

  ```yaml
  epoch: 1000 # 修改epoch
  LearningRate:
    base_lr: 0.0025 # 使用8个GPU进行分布式训练时，lr为0.02，在使用单个GPU时，应线性修改lr，故将0.02改为0.0025
    schedulers:
    - !PiecewiseDecay
      gamma: 0.1
      milestones: [300] # do not decay lr
    - !LinearWarmup
      start_factor: 0.3333333333333333
      steps: 1000
  ```


### 4. 训练及测试指令

##### 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/arsl/arsl_fcos_r50_fpn_coco_lxh.yml --eval
```

##### 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/rtdetr/rtdetr_r50vd_6x_coco_lxh.yml -o weights=outputlxh/rtdetr/model_final.pdparams
```

## 其他探索

### Semi-DETR

我们组尝试了一周 [Semi-DETR](https://github.com/JCZ404/Semi-DETR/tree/main) 的复现，但囿于单GPU的限制、项目默认的分布式训练方法和自身能力有限，以及项目的复杂性和独特性，涉及理论知识、算法实现、数据处理和优化策略等多个方面。因此，尽管有详细的文档和代码库可供参考，但从理论到实践的转化仍可能面临诸多难题。无法在指定时间内成功部署项目，将在今后努力尝试！也感谢助教姐姐耐心答疑，对我们组大有裨益！

### Transformer架构(YOLO-DETR)

与此同时，我们也探索了YOLO-DETR这样的Transformer架构，投入了相当的时间和精力。Transformer架构在物体检测领域的应用，如YOLO-DETR，涉及了多个复杂而又紧密相连的领域。从理论到实践的转化需要我们掌握Transformer的核心概念，同时还需要对目标检测任务有深刻的理解。我们需要熟悉处理图像数据、实现模型、调整超参数以及优化训练过程的技能。

但由于其模型复杂度过大，运行时间消耗太大，最终放弃了这个架构，选择了更低时间复杂度的实现方式。