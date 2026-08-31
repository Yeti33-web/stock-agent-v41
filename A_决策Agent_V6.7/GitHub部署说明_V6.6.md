# V6.6 GitHub部署说明

## 正式Agent A升级

1. 先备份当前GitHub仓库或保留旧版本压缩包。
2. 解压V6.6完整包。
3. 把解压后的文件上传到原GitHub仓库根目录，允许同名文件覆盖。
4. 必须同时覆盖整个`historical_test_tool`文件夹，保证Agent B动态读取最新A。
5. 等待Streamlit Community Cloud自动重启。

本次不需要重新运行SQL，不需要修改Streamlit Secrets，也不会删除Supabase账号、风险画像、会话、持仓或历史快照。

## 两个Streamlit应用入口

| 应用 | Main file path |
| --- | --- |
| 正式Agent A | `app.py` |
| 独立历史复现Agent B | `historical_test_tool/app.py` |

不要把A的入口改为B，也不要把B加入A的页面菜单。

## 部署后检查

1. A页面标题显示“融合校准版 V6.6”；
2. 查询一只股票后，观察分仍可显示；
3. 页面提示当前周期未通过跨股票样本外认证，不把分数解释为涨跌概率；
4. B页面的风险问卷、股票输入、持仓输入和结果标签与A保持一致；
5. B选择历史日期T后，结果显示实际采用交易日且不出现T之后行情；
6. B导出的JSON包含`Agent A逻辑身份`、版本、规则编号和SHA-256。

## 本地检查

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_v5.py test_v6.py test_v62.py test_v63.py test_v64.py test_v65.py
python -m unittest -v historical_test_tool.tests.test_historical_tool
```

本地启动A：

```bash
streamlit run app.py
```

本地启动B：

```bash
streamlit run historical_test_tool/app.py
```
