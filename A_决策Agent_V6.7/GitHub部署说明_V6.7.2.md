# V6.7.2 GitHub部署说明

## 本次新增（必须一起上传）

1. **`行业基准_V6.7.csv`** —— 行业标准化基准数据（A股33个核心行业）。**必须上传到仓库根目录**，
   否则净利率／负债率／PE 的行业标准化会自动回退为绝对阈值（不报错但新增因子失效）。
2. `生成行业基准.py` —— 需要刷新行业基准时在本地运行（约30分钟，可断点续跑）。
3. `完整因子字典_V6.7.2.csv`、`版本更新清单_V6.7.2.md` —— 说明文档，可不上传。
4. `test_v64.py`／`test_v65.py`／`test_v66.py` —— 更新后的自测脚本（96因子断言）。

## 升级步骤（沿用 V6.6 规则）

1. 解压 V6.7.2 完整包，把**所有同名文件**上传到原 GitHub 仓库根目录，**允许同名文件覆盖**；
2. **必须同时覆盖整个 `historical_test_tool` 文件夹**（Agent B 从仓库根目录动态读取最新 A 引擎）；
3. 确认 `行业基准_V6.7.csv` 出现在仓库根目录文件列表里；
4. 等待 Streamlit Community Cloud 自动重启。

## 部署后检查

- A 页面标题显示"融合校准与严格认证版 V6.7.2"；
- 查询 A 股后，基本面说明中出现"与同行业中位数比较"；
- B 页面顶部显示 `agent_a_version: V6.7.2`，`ruleset_id: fused-point-in-time-fundamental-v2`。

## 本地自测

```bash
python -m unittest -v test_v5.py test_v6.py test_v62.py test_v63.py test_v64.py test_v65.py test_v66.py
```
