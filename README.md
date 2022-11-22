# AlphX-Code-For-DAR
## 粤港澳大湾区（黄埔）国际算法算例大赛-古籍文档图像识别与分析算法比赛 AlphX队源码
[[说明文档](https://docs.qq.com/doc/DWk9IZ2JYVnNyc0hM)] [[PPT展示](https://docs.qq.com/doc/DWk9IZ2JYVnNyc0hM)]


我国的古籍文献资料记录承载着丰富的历史信息和文化传承，为响应古籍文化遗产保护的相关国家战略需求，古籍数字化工作势在必行。由于古籍文档图像存在版式复杂多变、不同朝代的刻字书写风格差异大等问题，古籍文档图像的分析于识别仍极具挑战。本方案整合现有优秀模型，实现汉文古籍文档图像的分析与识别。利用PAN++网络检测任意形状的文本列对象，并结合编码解码网络与启发式算法实现复杂页面的阅读顺序预测。根据前景像素比例，结合霍夫变换与上边缘对齐算法实现高效的任意形状文本串图像的扭曲倾斜矫正。针对过长的文字序列图像，使用叠瓦识别策略避免过度压缩导致的信息损失。最后，使用改进的卷积循环神经网络实现文本字符串图像的端到端识别。本地实验结果表明，所提出的方案稳定可靠，鲁棒性较高，在保持较高准确率的同时维持了合理的推理速度。 


![example](vis/image_553.jpg)
# 环境配置
`pip install -r requirements.txt`
# 训练
`
sh train.sh
`
# 推理
`
sh infer.sh
`
# 如有疑问，请随时通过ISSUE与我们取得联系

# 引用
如果您觉得我们的方案有一定帮助，请考虑引用如下工作~
### PAN++
```
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Zhibo, Yang and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
### 叠瓦识别策略
```
赵鹏海. 乌金体藏文文档版面分析与识别系统[D].西北民族大学,2022.DOI:10.27408/d.cnki.gxmzc.2022.000367.
```
