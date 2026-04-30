# 基于卷积神经网络的最小化白盒对抗攻击方法

本项目用于在 ImageNet 分类模型上运行白盒对抗攻击实验，并统计攻击成功率、SSIM、PSNR、修改像素比例等指标。原来的 Jupyter Notebook 已经工程化为 Python 脚本，后续实验直接用命令行运行。

## 环境安装

建议先创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 `torch` / `torchvision` 需要匹配特定 CUDA 版本，请先安装与本机 CUDA 对应的 PyTorch 版本，再执行：

```powershell
python -m pip install numpy Pillow scikit-image tqdm torchcam
```

## 文件说明

`attack_base.py`：所有攻击方法的基类，统一保存攻击名称、模型、设备，并定义 `forward` 接口。

`attack_utils.py`：实验公共工具，包括攻击成功判断、累计梯度剪枝、SSIM/PSNR/扰动统计、对抗样本保存和实验结果汇总。

`impl_pgd.py`：PGD 攻击实现。

`impl_mifgsm.py`：MI-FGSM 攻击实现。

`impl_cw.py`：CW Linf 风格攻击实现。

`impl_apgd_ce.py`：基于交叉熵损失的 APGD Linf 攻击实现。

`impl_apgd_dlr.py`：基于 DLR 损失的 APGD Linf 攻击实现。

`impl_myattack.py`：结合 ScoreCAM 区域约束和 PGD 的自定义攻击实现。

`experiment_runner.py`：统一实验运行模块，负责解析参数、加载模型、选择图片、创建输出目录、调用具体攻击类。

`run_attacks.py`：统一运行入口，可以通过 `--attack` 选择一个或多个攻击方法。

`run_pgd.py`、`run_mifgsm.py`、`run_cw.py`、`run_apgd_ce.py`、`run_apgd_dlr.py`、`run_mymodel.py`：单个攻击方法的便捷入口，内部复用 `experiment_runner.py`。

`image_process.py`：筛选 ImageNet 验证集中被指定模型正确分类的图片，并生成 `correct_1000_*` 数据目录和 metadata。

`cam_test.py`：生成 CAM 可视化结果，支持 ScoreCAM、GradCAM、GradCAM++、SmoothGradCAM++、LayerCAM。

`requirements.txt`：项目依赖列表。

`correct_1000_*`：各模型正确分类图片的数据目录，用作攻击实验输入。

`adversarial_samples/`：对抗样本输出目录。

## 运行实验

查看支持的攻击方法：

```powershell
python run_attacks.py --list-attacks
```

查看支持的模型：

```powershell
python run_attacks.py --list-models
```

运行单个攻击：

```powershell
python run_attacks.py --attack pgd --models resnet34 --selected-count 10
```

运行全部攻击：

```powershell
python run_attacks.py --attack all --models resnet34 --selected-count 10
```

运行多个模型：

```powershell
python run_attacks.py --attack pgd --models resnet34,vgg11,alexnet --selected-count 10
```

使用单独入口运行某个攻击：

```powershell
python run_pgd.py --models resnet34 --selected-count 10
python run_mifgsm.py --models resnet34 --selected-count 10
python run_cw.py --models resnet34 --selected-count 10
python run_apgd_ce.py --models resnet34 --selected-count 10
python run_apgd_dlr.py --models resnet34 --selected-count 10
python run_mymodel.py --models resnet34 --selected-count 10
```

常用参数示例：

```powershell
python run_attacks.py --attack pgd --eps 100/255 --alpha 1/255 --steps 100
python run_attacks.py --attack mifgsm --targeted --target-label 100
python run_attacks.py --attack pgd --no-targeted --no-save-adv
python run_attacks.py --attack mymodel --cam-keep-ratio 0.45 --prune
```

## 数据预处理

从 ImageNet 验证集筛选各模型预测正确的图片：

```powershell
python image_process.py --imagenet-dir E:\毕设\imagenet --output-root . --models all
```

只筛选某个模型：

```powershell
python image_process.py --imagenet-dir E:\毕设\imagenet --output-root . --models resnet34
```

## CAM 可视化

生成所有 CAM 方法的可视化：

```powershell
python cam_test.py --val-dir ILSVRC2012_img_val --output-dir ImageNet_CAM_swin --methods all
```

只生成 ScoreCAM：

```powershell
python cam_test.py --val-dir ILSVRC2012_img_val --methods scorecam --count 50
```
