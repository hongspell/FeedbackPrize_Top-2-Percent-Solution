# Feedback Prize - Evaluating Student Writing Top-2-Percent-Solution

[Feedback Prize - Evaluating Student Writing | Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021)

## 比赛介绍

在这次竞赛中，你将**识别学生写作中的元素**。更具体地说，你将自动分割文本，并对6-12年级学生所写的文章中的论证和修辞元素进行分类。帮助学生提高写作水平的一个方法是通过自动反馈工具，评估学生的写作并提供个性化的反馈。

- 竞赛类型：本次竞赛属于**深度学习/自然语言处理**，所以推荐使用的模型或者库：**Roberta/Deberta/Longformer**
- 赛题数据：官方提供的训练集大约有**15000篇文章**，测试集大约有**10000篇文章**。然后将分割的每个元素分类为以下内容之一：**引子/立场/主张/反诉/反驳/证据/结论性声明**，值得注意的是，文章的某些部分将是未加注释的（即它们不适合上述的分类）。
- 评估标准：**标签和预测的单词之间的重合度。**通过计算每个类别的TP/FP/FN，然后取所有类别的 **macro F1 score** 分数得出。 详见：[Feedback Prize - Evaluating Student Writing | Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021/overview/evaluation)
- 推荐阅读 Kaggle 内的一篇 EDA（探索数据分析）来获取一些预备知识：[NLP on Student Writing: EDA | Kaggle](https://www.kaggle.com/code/erikbruin/nlp-on-student-writing-eda)



## 数据说明

官方提供的训练集大约有**15000篇文章**，测试集大约有**10000篇文章**。然后将分割的每个元素分类为以下内容之一：**引子/立场/主张/反诉/反驳/证据/结论性声明**，值得注意的是，文章的某些部分将是未加注释的（即它们不适合上述的分类）。

官方数据页面 [Feedback Prize - Evaluating Student Writing | Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021/data)

- 将分割的每个元素分类为以下内容之一:

  **引子**--以统计数字、引文、描述或其他一些手段开始的介绍，以吸引读者的注意力并指向论题

  **立场**--对主要问题的看法或结论

  **主张**--支持该立场的主张

  **反诉**--反驳另一个诉求的诉求，或提出与立场相反的理由

  **反驳**--驳斥反诉的主张

  **证据**--支持主张、反主张或反驳的观点或例子。

  **结论性声明**--重申主张的结论性声明

  值得注意的是，**文章的某些部分将是未加注释的（即它们不适合上述的分类）。**

  

- train.csv - 一个包含训练集中所有论文注释版本的.csv文件

  **id** - 作文的ID

  **discourse_id** - 话语元素的ID

  **discourse_start** - 话语元素在文章中开始的字符位置

  **discourse_end** - 话语元素在文章中结束的位置

  **discourse_text** - 话语元素的文本

  **discourse_type** - 话语元素的分类

  **discourse_type_num** - 话语元素的列举式分类标签（带序号）

  **predictionstring** - 训练样本的词索引，为预测所需。




## 解决方案思路
本次竞赛我们的方案采用了 **longformer+ deberta **的双模型融合，由于官方数据的有一些不干净的原标签，所以我们使用经过修复的corrected_train.csv（已下载到目录下）。

在文本数据的处理上，我们将**max_len**设置在了1024（在推理是扩大至longformer=4096/deberta=2048）。之后我们对数据做了**10Fold**的标准切分。

模型上我们选择了**allenai/longformer-base-4096**和**microsoft/deberta-large**版本，在之后接了一个Dropout层和Linear层。

在模型预测出结果后，我们使用了**后处理**的方式来进一步筛选预测的实体，主要是对**每种实体的最小长度**和**最小置信度**做出限制，如果小于阈值则被后处理筛掉。

#### 模型代码

```python
class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        # 载入 backbone
        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels) # 分类头
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        # 五个不同的dropout结果
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5 # 五层取平均
        return logits
```



#### 后处理参数

```python
# 每种实体的的最小长度阈值，小于阈值不识别
MIN_THRESH = {
    "I-Lead": 11,
    "I-Position": 7,
    "I-Evidence": 12,
    "I-Claim": 1,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

# 每种实体的的最小置信度，小于阈值不识别
PROB_THRESH = {
    "I-Lead": 0.687,
    "I-Position": 0.537,
    "I-Evidence": 0.637,
    "I-Claim": 0.537,
    "I-Concluding Statement": 0.687,
    "I-Counterclaim": 0.37,
    "I-Rebuttal": 0.537,
}
```



## 比赛上分历程

1. longformer Baseline 5Fold，Public LB : 0.678；
2. 将推理阶段的max_len设置为4096，Public LB : 0.688；
3. 加入后处理，Public LB : 0.694；
4. 尝试了deberta-base 但分数太低，我们没有尝试将其加入融合；
5. deberta-large 5Fold 加入后处理，Public LB : 0.705；
6. 将两个模型融合，Public LB：0.709；
7. 对学习率，epoch等进行调参，Public LB：0.712；
8. 使用修复标签后的corrected_train.csv，Public LB：0.714；
9. 尝试将5fold换成10fold，Public LB：0.716；
10. 对后处理进行调参，Public LB：0.718；



## 代码、数据集

+ 代码
  + feedback_train.ipynb
  + feedback_inference.ipynb
  
+ 数据集

  + 官方数据集

  + corrected_train.csv

  + https://www.kaggle.com/xhlulu/deberta

  + https://www.kaggle.com/abhishek/tez-lib

    


## TL;DR

竞赛是由乔治亚州立大学举办的，对学生写作中的论证和修辞元素进行识别。本次竞赛在数据上我们修复了官方数据的不干净的原标签部分，整体的方案上采用了 **longformer+ deberta **的双模型融合，为了防止过拟合我们尝试了在模型头部位置加入Dropout层。我们训练出了**longformer-base-4096**和**deberta-large**两个模型，再通过后处理对**每种实体的最小长度**和**最小置信度**做出限制，筛掉小于阈值的预测值，最后进行CV-10Fold和简单的加权融合。此外，我们还尝试了deberta-base等，但没有起效果。最终我们获得了Private LB: 0.718 (Top2%) 的成绩。

