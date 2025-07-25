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

#### 1.1.2 图像预处理，lora权重V1，自定义参数，SmolVLM-Instruct，prompt1

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

#### 1.1.3 图像预处理，lora权重V1，自定义参数，SmolVLM-Instruct，prompt2

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
| **blue_on_green**  |      88      |
|  **blue_on_red**   |      96      |
| **green_and_blue** |     100      |
| **green_on_blue**  |      82      |
|  **green_on_red**  |      90      |
|  **red_and_blue**  |     100      |
| **red_and_green**  |     100      |
|  **red_on_blue**   |      88      |
|  **red_on_green**  |      94      |

没有任何变化，一模一样

jsonl文件：

### 1.2 基于每个类200张的完整数据集

#### 1.2.1 图像预处理，无lora权重，自定义参数，SmolVLM-Instruct，prompt2

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
| **blue_on_green**  |      89      |
|  **blue_on_red**   |      88      |
| **green_and_blue** |     79.5     |
| **green_on_blue**  |              |
|  **green_on_red**  |              |
|  **red_and_blue**  |              |
| **red_and_green**  |              |
|  **red_on_blue**   |              |
|  **red_on_green**  |              |

性能不及之前，需要加Lora权重

#### 1.2.2 图像预处理，lora权重V1，自定义参数，SmolVLM-Instruct，prompt2，TTA

加入了TTA逻辑

|                    | **Accuracy**_no_TTA | **Accuracy**_TTA |
| :----------------: | :-----------------: | :--------------: |
|      **red**       |                     |                  |
|     **green**      |                     |                  |
|      **blue**      |                     |                  |
|      **wood**      |                     |                  |
| **blue_on_green**  |         88          |        91        |
|  **blue_on_red**   |         96          |        94        |
| **green_and_blue** |         100         |       100        |
| **green_on_blue**  |         82          |       87.5       |
|  **green_on_red**  |         90          |       91.5       |
|  **red_and_blue**  |         100         |       100        |
| **red_and_green**  |         100         |       100        |
|  **red_on_blue**   |         88          |        90        |
|  **red_on_green**  |         94          |        91        |

#### 1.2.4 图像预处理，lora权重V5，自定义参数，SmolVLM-Instruct，prompt2

|                    | **Accuracy** |
| :----------------: | :----------: |
|      **red**       |              |
|     **green**      |              |
|      **blue**      |              |
|      **wood**      |              |
| **blue_on_green**  |      84      |
|  **blue_on_red**   |     96.5     |
| **green_and_blue** |     94.5     |
| **green_on_blue**  |     85.5     |
|  **green_on_red**  |     90.5     |
|  **red_and_blue**  |     100      |
| **red_and_green**  |      99      |
|  **red_on_blue**   |     91.5     |
|  **red_on_green**  |     93.5     |



## 2. 目标数量识别(3个方块)

### 2.2 基于每个类200张的数据集

#### 2.2.1 图像预处理，初始权重，自定义参数，SmolVLM-Instruct，prompt2

```
prompt = "Think step by step, but only answer with a number: How many objects are there in the image?"
```

**Acc1**

#### 2.2.2 图像预处理，lora权重V1，自定义参数，SmolVLM-Instruct，prompt2

```
prompt = "Think step by step, but only answer with a number: How many objects are there in the image?"
```

**Acc2**

#### 2.2.3 图像预处理，lora权重V1，自定义参数，SmolVLM-Instruct，prompt2，TTA

```
prompt = "Think step by step, but only answer with a number: How many objects are there in the image?"
```

使用TTA技术

**Acc3**

检测发现，效果并不好

#### 2.2.4 检测结果

**2+1摆放模式**

|                                 |                             img                              | Acc1 | **Acc2** | Acc3 |
| :-----------------------------: | :----------------------------------------------------------: | ---- | :------: | ---- |
| blue_on_green_and_red_separated | ![blue_on_green_and_red_separated_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\blue_on_green_and_red_separated_0001.jpg) | 100  |   100    | 100  |
| blue_on_red_and_green_separated | ![blue_on_red_and_green_separated_0200](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\blue_on_red_and_green_separated_0200.jpg) | 98.5 |    97    | 94   |
| green_on_blue_and_red_separated | ![green_on_blue_and_red_separated_0200](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\green_on_blue_and_red_separated_0200.jpg) | 79.5 |    75    | 69.5 |
| green_on_red_and_blue_separated | ![green_on_red_and_blue_separated_0200](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\green_on_red_and_blue_separated_0200.jpg) | 100  |   100    | 100  |
| red_on_blue_and_green_separated | ![red_on_blue_and_green_separated_0200](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\red_on_blue_and_green_separated_0200.jpg) | 75   |    79    | 75.5 |
| red_on_green_and_blue_separated | ![red_on_green_and_blue_separated_0200](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\red_on_green_and_blue_separated_0200.jpg) | 100  |   100    | 100  |

错误的回答几乎都是2个方块

**金字塔摆放**

|                               |                                                              | Acc1 | Acc2 |
| ----------------------------- | ------------------------------------------------------------ | ---- | :--: |
| pyramid_blue_on_green_and_red | ![pyramid_blue_on_green_and_red_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_blue_on_green_and_red_0001.jpg) | 100  | 100  |
| pyramid_blue_on_red_and_green | ![pyramid_blue_on_red_and_green_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_blue_on_red_and_green_0001.jpg) | 96   |  98  |
| pyramid_green_on_blue_and_red | ![pyramid_green_on_blue_and_red_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_green_on_blue_and_red_0001.jpg) | 97   | 97.5 |
| pyramid_green_on_red_and_blue | ![pyramid_green_on_red_and_blue_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_green_on_red_and_blue_0001.jpg) | 94   |  94  |
| pyramid_red_on_blue_and_green | ![pyramid_red_on_blue_and_green_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_red_on_blue_and_green_0001.jpg) | 100  | 100  |
| pyramid_red_on_green_and_blue | ![pyramid_red_on_green_and_blue_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\pyramid_red_on_green_and_blue_0001.jpg) | 99.5 | 98.5 |

错误的回答几乎都是4个方块

**上下堆叠**

|                      |                                                              | Acc1 | Acc2 |
| :------------------: | :----------------------------------------------------------: | ---- | :--: |
| blue_on_green_on_red | ![blue_on_green_on_red_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\blue_on_green_on_red_0001.jpg) | 98.5 | 98.5 |
| blue_on_red_on_green | ![blue_on_red_on_green_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\blue_on_red_on_green_0001.jpg) | 95   | 95.5 |
| green_on_blue_on_red | ![green_on_blue_on_red_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\green_on_blue_on_red_0001.jpg) | 98   |  98  |
| green_on_red_on_blue | ![green_on_red_on_blue_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\green_on_red_on_blue_0001.jpg) | 95.5 | 96.5 |
| red_on_blue_on_green | ![red_on_blue_on_green_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\red_on_blue_on_green_0001.jpg) | 98.5 |  99  |
| red_on_green_on_blue | ![red_on_green_on_blue_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\red_on_green_on_blue_0001.jpg) | 98   |  99  |

错误的回答几乎都是4个方块

**分开摆放**

|                        |                                                              | Acc1 | Acc2 |
| ---------------------- | ------------------------------------------------------------ | ---- | :--: |
| red_and_green_and_blue | ![red_and_green_and_blue_0001](E:\LLMs\SmolVLM_Version3\image_test_batch\image_3obj\red_and_green_and_blue_0001.jpg) | 100  | 100  |



## 3. 两个方块，相对位置关系识别

### 3.1 基于每个类50张的init数据集

#### 3.1.1 图像预处理，lora权重V1，自定义参数，prompt1，SmolVLM-Instruct

```
prompt = "This image shows the workspace before a robot arm performs a grasping task. There are exactly two objects in the workspace. Please describe the color of each object and their spatial relationship.(for example, whether they are stacked or separated, and which one is on top if they are stacked)."
```

先使用这个提示词，然后再使用正则化脚本结合llm，直接提取出结构关系，例子如下：

```
"parsed": {"relationship": "stacked", "top": {"color": "blue"}, "bottom": {"color": "green"}}
```

|                    | **Accuracy** |
| :----------------: | :----------: |
| **blue_on_green**  |      98      |
|  **blue_on_red**   |      98      |
| **green_and_blue** |      94      |
| **green_on_blue**  |      96      |
|  **green_on_red**  |      92      |
|  **red_and_blue**  |      96      |
| **red_and_green**  |      76      |
|  **red_on_blue**   |      96      |
|  **red_on_green**  |     100      |

#### 3.1.2 图像预处理，lora权重V1，自定义参数，prompt2，SmolVLM-Instruct

```
prompt = "This image shows the workspace before a robot arm performs a grasping task. There are exactly two objects! Please describe the color of each object and their spatial relationship.(for example, whether they are stacked or separated, and which one is on top if they are stacked)."
```

先使用这个提示词，然后再使用正则化脚本结合llm，直接提取出结构关系，例子如下：

```
"parsed": {"relationship": "stacked", "top": {"color": "blue"}, "bottom": {"color": "green"}}
```

|                    | **Accuracy** |
| :----------------: | :----------: |
| **blue_on_green**  |      94      |
|  **blue_on_red**   |     100      |
| **green_and_blue** |      94      |
| **green_on_blue**  |      98      |
|  **green_on_red**  |      90      |
|  **red_and_blue**  |      92      |
| **red_and_green**  |      78      |
|  **red_on_blue**   |      98      |
|  **red_on_green**  |     100      |
