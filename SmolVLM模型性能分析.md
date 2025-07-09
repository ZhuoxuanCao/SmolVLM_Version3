# SmolVLM模型性能分析-Version3

在这个项目文件中，我们更新了两个方块上下堆叠的数据集。

## 1. 目标数量识别(1个&2个方块)

### 1.1 基于每个类50张的init数据集

#### 1.1.1 无图像预处理，无lora权重，默认参数，SmolVLM-Instruct，prompt1

```
 prompt = "How many objects are there in the image? Only answer with a number."
```

|                    | **Accuracy** |
| :----------------: | :----------: |
|      **red**       |     100      |
|     **green**      |     100      |
|      **blue**      |     100      |
|      **wood**      |     100      |
| **blue_on_green**  |      60      |
|  **blue_on_red**   |      62      |
| **green_and_blue** |     100      |
| **green_on_blue**  |      70      |
|  **green_on_red**  |      70      |
|  **red_and_blue**  |     100      |
| **red_and_green**  |     100      |
|  **red_on_blue**   |      70      |
|  **red_on_green**  |      60      |

可以看出，单个方块识别已经完美，但是当方块有上下堆叠时，识别准确率极低

jsonl文件：

```
./predictions_1cube/num_obj/predictions_img_init_1obj_v1.jsonl

./predictions_2cube/num_obj/predictions_img_init_2obj_v1.jsonl
```

#### 1.1.2 图像预处理，lora权重，自定义参数，SmolVLM-Instruct，prompt1

```
 prompt = "How many objects are there in the image? Only answer with a number."
```

|                    | **Accuracy** |
| :----------------: | :----------: |
|      **red**       |     100      |
|     **green**      |     100      |
|      **blue**      |     100      |
|      **wood**      |     100      |
| **blue_on_green**  |    **88**    |
|  **blue_on_red**   |      96      |
| **green_and_blue** |     100      |
| **green_on_blue**  |    **82**    |
|  **green_on_red**  |      90      |
|  **red_and_blue**  |     100      |
| **red_and_green**  |     100      |
|  **red_on_blue**   |    **88**    |
|  **red_on_green**  |      94      |

在单方块和并排摆放的双方块场景下，数量识别准确率可达100%，但在堆叠（上下遮挡）场景下，尤其是涉及绿色和蓝色的组合时，模型准确率明显下降（最低仅82%）。

这一现象表明，当前多模态模型在处理部分遮挡物体的计数任务时仍有显著的改进空间。建议在数据集扩充、数据增强及模型视觉感知能力提升等方面进一步优化，提升模型对复杂空间关系和遮挡场景的泛化能力。

jsonl文件：

```
./predictions_1cube/num_obj/predictions_img_init_1obj_v2.jsonl

./predictions_2cube/num_obj/predictions_img_init_2obj_v2.jsonl
```

#### 1.1.3 图像预处理，lora权重，自定义参数，SmolVLM-Instruct，prompt2

```
prompt = "Think step by step, but only answer with a number: How many objects are there in the image?"
```

**Chain-of-Thought-then-Strip**

让模型先用一步步推理方式考虑，但最后**只输出数字**。比如 prompt 变成：“Think step by step but output only the final number.”
 这种做法能让模型更谨慎考虑“场景里有没有被遮挡的方块”，通常会更少犯低级错误。

|                    | **Accuracy** |
| :----------------: | :----------: |
|      **red**       |              |
|     **green**      |              |
|      **blue**      |              |
|      **wood**      |              |
| **blue_on_green**  |              |
|  **blue_on_red**   |              |
| **green_and_blue** |              |
| **green_on_blue**  |              |
|  **green_on_red**  |              |
|  **red_and_blue**  |              |
| **red_and_green**  |              |
|  **red_on_blue**   |              |
|  **red_on_green**  |              |

jsonl文件：



## 2. 两个方块，堆叠与分开摆放的识别

