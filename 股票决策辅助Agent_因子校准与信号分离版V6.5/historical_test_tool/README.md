# 股票 Agent 完整界面独立历史时点测试工具 V2.1

这个工具用于回答：

> 假设现在是历史日期 T，V6.5 股票 Agent 当时会给出什么风险等级、评分、方向可信度、持有周期和结论？

## 和原 Agent 相同的内容

- 同一套 14 题个人风险测评；
- 同样的 A股、美股、港股输入；
- 同样的是否持有、持股数量／持仓金额、成本、加仓金额、杠杆输入；
- 同样的因子、公式、权重、风险阈值和结论规则；
- 同样的结果标签：结论、卖出信号、相似周期预测、最新资讯、风险与仓位、持有周期、因子解释与验证、数据证据、专业指标。

结果页面直接复用当前 V6.5 `app.py` 中的原展示函数，不另写一套简化结果。

V2.1会主动忽略正式 `app.py` 中可能残留的旧版历史测试导入，避免新旧测试文件互相调用。

## 唯一的分析差别

页面会多填一个“历史分析日期 T”。运行时系统假设今天就是 T：

1. 如果 T 不是交易日，采用 T 之前最近一个交易日；
2. 行情接口只请求 T 日及以前约五年的数据；
3. 进入 Agent 前再次删除并检查 T 日之后的记录；
4. 只把在 T 日能够安全确认的数据传给 V6.5 分析逻辑；
5. 输出 Agent 当时的完整判断，到此结束。

本工具没有“验证期限 H”，也不读取或评价 T 日之后的实际涨跌。之后看多少天，应以 Agent 自动给出的持有／复核周期为准，再由测试人员人工核对真实行情。

## 为什么部分历史证据会显示“未参与”

免费数据源目前没有统一可靠的历史披露快照。因此：

- 股票、成交量和市场基准：使用 T 日及以前的数据；
- 财务数据：没有可核验实际披露日的历史快照时不参与；
- 历史利率：没有完成逐发布日期核验时不参与；
- 历史资讯：没有可核验历史发布时间的数据源时不参与。

这样做是为了避免把今天才知道的信息倒填到过去。缺失项会沿用原 Agent 的缺失数据处理规则，并降低数据完整度，不会伪造数据。

## 为什么它仍与正式 Agent 分开

- `historical_test_tool/app.py` 是独立入口；
- 不修改正式 `app.py`；
- 不增加正式 Agent 菜单；
- 不写入正式邮箱账号、风险资料、持仓、股票会话或数据库；
- 不因测试结果自动修改因子、权重或阈值；
- “今天=T”只在本次测试进程中临时生效，结束后自动恢复。

因此，风险测评和分析输入／输出与原 Agent 保持一致，但测试数据不会污染正式用户记录。

## 放入 GitHub

把整个 `historical_test_tool` 文件夹上传到 V6.5 仓库根目录。上传后目录应类似：

```text
你的仓库/
├── app.py                         # 原正式 Agent，不修改
├── agent_core.py                  # 原分析逻辑，不修改
├── factor_analysis.py             # 原因子逻辑，不修改
├── questionnaire.py               # 原风险问卷，不修改
├── requirements.txt
└── historical_test_tool/          # 新增独立测试工具
    ├── app.py
    ├── original_ui.py
    ├── historical_data.py
    ├── point_in_time.py
    ├── runner.py
    └── tests/
```

## 本地启动

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
streamlit run historical_test_tool/app.py
```

## Streamlit Community Cloud 独立部署

新建一个独立应用，并选择：

- Repository：与正式 Agent 相同的 GitHub 仓库；
- Branch：存放测试工具的分支；
- Main file path：`historical_test_tool/app.py`。

不要把正式应用的 Main file path 从 `app.py` 改掉。正式 Agent 和测试 Agent 应是两个不同的 Streamlit 应用。

## 自动检查

在仓库根目录运行：

```bash
python -m unittest historical_test_tool.tests.test_historical_tool -v
```

检查内容包括：非交易日回退、T 后数据阻断、临时日期恢复、用户实际风险画像参与计算，以及 V6.5 结果页面复用。
