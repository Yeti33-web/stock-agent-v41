# GitHub覆盖步骤

1. 解压下载的压缩包。
2. 打开里面的 `Agent_A_正式版_V6.5.2` 文件夹。
3. 进入Agent A的GitHub仓库根目录，也就是当前能看到旧 `app.py` 的位置。
4. 点击 **Add file → Upload files**。
5. 将 `Agent_A_正式版_V6.5.2` 文件夹里面的全部内容拖进去，不要把外层文件夹整体套进仓库。
6. 提交后等待Streamlit自动重启。
7. 打开网页，标题必须显示：`数据渠道稳定版 V6.5.2`。
8. Streamlit的Main file path仍然是 `app.py`。

不需要重新运行SQL，不需要修改Streamlit Secrets，不会删除账号、问卷、会话或持仓记录。

如果网页标题仍是V6.5、V6.6或V6.7，说明文件没有覆盖到仓库根目录。

