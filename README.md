# 注意力演化图谱

[打开交互教学](https://johanlee65535-verp2.github.io/attention-atlas/)

同一份 **8 × 512** 合成输入贯穿七章。

1. 普通 Attention 的因果打分、Softmax 和加权求和
2. MHA 的拆头、并行计算和拼接
3. 线性 Attention 的核化、固定尺寸状态和归一化
4. Delta Rule 的预测误差和定向写入
5. Gated DeltaNet 的逐头标量衰减
6. GVA 中的 Q/K 共享和独立 value 状态
7. Kimi Delta Attention 的逐通道衰减

点选 token 或拖动进度条可定位到任意一步。MHA 和 GVA 可以切换头，Delta、GDN 和 KDA 可以调节相应的门值。单头宽度为 64，页面展示的 4×4 状态切片来自完整的 64×64 计算。

本地可直接打开 `index.html`。页面无需构建或安装依赖。论文链接和教学简化说明在页面底部。
