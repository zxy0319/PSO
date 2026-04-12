# PSO 编码解码模拟器

这是一个用于半导体晶圆制造的 PSO (Particle Swarm Optimization) 编码解码规则实现。

## 功能特性

### 编码 (Encoding)
- 输入：2N 维连续向量 (0-1 之间)，N 为晶圆数量
- 前 N 个基因：批处理优先级 (batch priorities)
- 后 N 个基因：抢占优先级 (preemptive priorities)

### 解码 (Decoding)
- 事件驱动模拟器：模拟晶圆在 9 道工序中的流动
- 标准机：根据抢占优先级选择最高优先级晶圆
- 批处理机：根据批处理优先级排序后 First-Fit 装批
- 自动修复约束违反

### 分时电价成本计算 (TOU Energy Cost)
- 基于机器功率和分时电价表计算总能源成本
- 电价表：
  - 谷时 (0-6): 0.4 元/kWh
  - 平时 (6-8, 12-18, 22-24): 0.8 元/kWh
  - 峰时 (8-12, 18-22): 1.2 元/kWh
- **修复**: 批处理机能耗不再重复计费（基于机器历史而非工件历史）
- **修复**: 时间边界计算使用 math.floor 确保准确跨小时积分

## PSO 优化器

### 实现特性
- 多目标优化：同时优化 makespan 和总能源成本
- 可配置权重：makespan_weight 和 tec_weight
- 标准PSO参数：惯性权重 w=0.8，认知系数 c1=2.0，社会系数 c2=2.0

### 使用方法

```python
from pso_optimizer import PSOConfig, PSOOptimizer
from pso_encoding_decoding import PSOSimulator, build_default_factory_config

# 配置PSO
config = PSOConfig(
    num_particles=50,
    max_iterations=100,
    num_wafers=5,
    makespan_weight=0.6,  # 更重视makespan
    tec_weight=0.4
)

# 运行优化
simulator = PSOSimulator(build_default_factory_config(batch_capacity=4))
optimizer = PSOOptimizer(config, simulator)
best_position, best_result, fitness_history = optimizer.optimize()

print(f"最佳makespan: {best_result.makespan}")
print(f"最佳能源成本: {best_result.total_energy_cost}")
```

## 实验结果

### 随机搜索基准测试
- 测试了500个随机解
- 所有解产生相同的 makespan = 269.00 小时
- 所有解产生相同的 TEC = 4816.00 元
- 这表明当前编码解码机制对这个特定配置的鲁棒性有限

### PSO优化结果
- 在相同配置下，PSO找到的解与随机搜索一致
- 没有发现更好的解，可能是因为：
  1. 搜索空间中只有一个可行解
  2. 编码方案需要改进
  3. 需要更大的晶圆数量或更复杂的工厂配置

## 扩展方向

1. **增加晶圆数量**：测试更大规模的问题
2. **修改工厂配置**：改变机器数量、处理时间或批容量
3. **改进编码方案**：使用不同的优先级编码方式
4. **多目标优化**：实现真正的Pareto优化而非加权和
5. **约束处理**：添加更多现实世界的约束条件

## 文件结构

- `pso_encoding_decoding.py`: 核心编码解码模拟器
- `pso_optimizer.py`: PSO优化算法实现
- `comparison.py`: 优化方法对比分析
- `README.md`: 项目文档