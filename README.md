# 医学髋关节 X 光图像关键点检测框架

本项目用于搭建髋关节发育不良（DDH）关键点检测第一阶段的基础框架。

## 目标
- 10个关键点自动检测：`TeardropR、TeardropL、TiR、TiL、FHR、FHL、tonnisR1、tonnisR2、tonnisL1、tonnisL2`
- 支持 `labelme` 标注文件解析
- 支持训练、验证、测试预测
- 支持遮盖敏感信息的区域涂黑预处理

## 目录结构

```
MSSB/
  requirements.txt
  README.md
  .gitignore
  src/
    dataset.py
    model.py
    train.py
    predict.py
    utils.py
  data/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
```

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 数据组织

- `data/train/images/`：训练图像
- `data/train/labels/`：训练图像对应的 `labelme` JSON 标注
- `data/val/images/`：验证图像
- `data/val/labels/`：验证图像对应的 `labelme` JSON 标注
- `data/test/images/`：测试图像（仅用于最终评估，不参与训练）

> 请不要重新划分训练/验证/测试集，测试集仅用于推理。

## 运行训练

```bash
python src/train.py \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir outputs \
  --epochs 20 \
  --batch-size 8
```

## 运行预测

```bash
python src/predict.py \
  --test-dir data/test \
  --checkpoint outputs/checkpoint.pt \
  --output-file outputs/test_predictions.csv
```

## 数据标注流程（labelme）

1. 使用 `labelme` 打开训练/验证图像
2. 标注 10 个关键点，名称必须与下列名称一致：
   - `TeardropR`
   - `TeardropL`
   - `TiR`
   - `TiL`
   - `FHR`
   - `FHL`
   - `tonnisR1`
   - `tonnisR2`
   - `tonnisL1`
   - `tonnisL2`
3. 保存 JSON 标注文件到 `data/<split>/labels/`

## 提示

- 可在训练前使用 `src/utils.py` 中的 `mask_sensitive_region` 对图像左上角进行涂黑处理
- 若要进行数据增强，可在 `src/dataset.py` 的 `build_transforms` 中扩展
- 若使用额外数据或预训练模型，请在报告中说明来源和使用方式
