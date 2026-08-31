# V6.5 GitHub 部署说明

## 升级方式

1. 先下载并解压V6.5压缩包。
2. 打开原Agent的GitHub仓库根目录。
3. 上传并允许覆盖：
   - `agent_core.py`
   - `factor_analysis.py`
   - `add_position_analysis.py`
   - `news_analysis.py`
   - `app.py`
   - `README.md`
4. 新增：
   - `test_v65.py`
   - `因子校准说明_V6.5.md`
   - `版本更新清单_V6.5.md`
   - `GitHub部署说明_V6.5.md`
5. 如果你也希望历史复现工具读取V6.5逻辑，再覆盖整个 `historical_test_tool` 文件夹。
6. 等待Streamlit Community Cloud自动重启。

## 不需要操作

- 不需要重新运行SQL。
- 不需要修改Streamlit Secrets。
- 不需要删除Supabase账号、风险画像、股票会话或持仓数据。
- 不要删除仓库里的其他旧文件。

## 部署后检查

1. 页面标题是“因子校准与信号分离版 V6.5”。
2. 完成风险测评后查询一只股票。
3. 结果顶部应分开显示：用户风险、股票风险、行情方向、方向可信度、个人适配、数据完整度。
4. 如历史验证未通过，页面应显示“历史验证未通过／不判断方向”，而不是强行给出买入判断。
5. 打开加仓页，确认方向验证未通过时不允许加仓。

建议保留V6.4压缩包作为旧版备份，方便对比同一历史时点的输出差异。

