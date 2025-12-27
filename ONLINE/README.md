# Real-time Digital Twin System - 5DOF Model

实时数字孪生系统，用于5自由度结构模型的实时仿真和监测。

## 架构

系统分为四个独立模块：

- **INPUT**: 力生成器 - 生成白噪音 + 周期性脉冲荷载
- **SYSTEM**: 系统参数提供者 - 生成 M(t), K(t), C(t) 矩阵
- **OUTPUT**: 响应求解器 - 使用 Newmark-Beta 方法计算结构响应
- **DASHBOARD**: Web 可视化界面 - 实时显示结构状态

## 安装

```bash
cd ONLINE
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

然后打开浏览器访问: http://127.0.0.1:5000

## 配置

系统参数可在 `config/system_config.py` 中修改：
- 自由度数量 (nDOF)
- 时间步长 (dt)
- 初始质量和刚度
- 阻尼比

## 模块说明

### INPUT 模块
- 每个自由度添加白噪音
- 每10秒在随机自由度上生成短脉冲荷载

### SYSTEM 模块
- 可配置为固定参数或时变参数
- 生成质量矩阵 M、刚度矩阵 K、阻尼矩阵 C

### OUTPUT 模块
- 实时 Newmark-Beta 求解器
- 根据 t-1 时刻状态和 t 时刻输入计算响应

### DASHBOARD 模块
- 3D 糖葫芦串模型可视化
- 实时位移、速度、加速度显示
- 位移历史曲线

