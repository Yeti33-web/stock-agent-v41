# 历史部署说明（V6.4）

> 当前版本是V6.6。请优先按 [GitHub部署说明_V6.6.md](GitHub部署说明_V6.6.md) 操作；以下内容只保留作为V6.4历史记录。

## 已经正常运行 V6.3 的用户：只做下面三步

### 第一步：上传 GitHub 中六个新版文件

1. 打开原 GitHub 仓库 `stock-agent-v41`。
2. 点击 **Add file → Upload files**。
3. 从 V6.4 文件夹选择：
   - `app.py`
   - `README.md`
   - `factor_analysis.py`
   - `因子字典与历史验证_V6.4.md`
   - `版本更新清单_V6.4.md`
   - `test_v64.py`
4. 保持 **Commit directly to the main branch**，点击 **Commit changes**。

`app.py` 和 `README.md` 会覆盖旧文件，其余是新增文件。不要删除或改名其他文件。

### 第二步：等待网页自动更新

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 找到原股票 Agent，等待自动重新部署。
3. 如果几分钟后仍显示 V6.3，点击应用右侧 `⋮` → **Reboot app**。

本次不需要重新运行 SQL，不需要修改 Streamlit Secrets，也不需要重新设置验证码邮件或 SMTP；已有云端数据不会被清除。

### 第三步：测试因子解释与历史验证

1. 打开原 `.streamlit.app` 网址并登录。
2. 重新分析任意一只 A股、美股或港股。
3. 打开结果页的 **因子解释与验证**。
4. 确认能看到本次贡献图、用户风险分、股票风险分和历史验证表。
5. 展开完整因子字典，确认可以按模块筛选。
6. 再打开原有 **最新资讯、卖出信号、加仓适配分析和股票会话**，确认原功能仍正常。

---

## 尚未部署过永久版的用户：完整部署步骤

## 第一步：上传 GitHub 文件

1. 打开你原来的 GitHub 仓库 `stock-agent-v41`。
2. 点击 **Add file → Upload files**。
3. 从解压后的 V6.4 文件夹选择并上传以下文件：
   - `app.py`
   - `add_position_analysis.py`
   - `news_analysis.py`
   - `factor_analysis.py`
   - `agent_core.py`
   - `questionnaire.py`
   - `session_memory.py`
   - `cloud_store.py`
   - `snapshot_codec.py`
   - `requirements.txt`
   - `一键建表_V6.0.sql`
   - `README.md`
   - `因子字典与历史验证_V6.4.md`
   - `版本更新清单_V6.4.md`
   - `GitHub部署说明.md`
   - `Supabase注册验证码邮件模板.html`
   - `.gitignore`（如果浏览器不显示隐藏文件，可不上传）
4. 保持 **Commit directly to the main branch**，点击 **Commit changes**。

注意：同名的 `app.py`、`agent_core.py` 等必须覆盖旧版，不能改名，也不要多套一层文件夹。

## 第二步：在 Supabase 运行一次建表脚本

1. 打开 [Supabase Dashboard](https://supabase.com/dashboard)，进入你的 `stock-agent-v5` 项目。
2. 点击左侧 **SQL Editor**。
3. 点击 **New query**。
4. 用记事本打开解压文件夹内的 `一键建表_V6.0.sql`，全选并复制。
5. 粘贴到 SQL Editor，点击右上角绿色 **Run**。
6. 下方看到 `V6.0 数据表、自动持仓汇总和用户隔离权限已配置完成` 即成功。

这一步只需运行一次。脚本可重复运行，不会主动删除旧数据。

## 第三步：复制两个公开连接参数

1. 在 Supabase 项目首页复制 **Project URL**。
2. 点击顶部 **Connect**，找到 API Keys。
3. 复制 **publishable key**。

只复制 publishable key。不要复制 secret key、service key 或数据库密码。

## 第四步：填写 Streamlit Secrets

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 在 **My apps** 找到你的股票 Agent。
3. 点击应用右侧 `⋮` → **Settings** → **Secrets**。
4. 粘贴下面两行，并把引号内内容换成你刚才复制的真实值：

```toml
SUPABASE_URL = "https://你的项目编号.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "你的 publishable key"
```

5. 点击 **Save**，再点击 **Reboot app**；如果没有 Reboot 按钮，等待应用自动重启。

## 第五步：设置验证码邮件并检查永久保存

1. 先按本说明上方“第二步：把注册邮件改为验证码”设置 Confirm signup 邮件模板。
2. 打开原来的 `.streamlit.app` 网址。
3. 选择 **首次注册**，填写邮箱和至少8位密码，发送验证码。
4. 回到同一网页输入邮件中的验证码，完成验证并自动登录。
5. 完成风险测评，分析一只股票，并在股票会话中添加一条备注。
6. 用手机或另一台电脑打开同一网址，以同一邮箱和密码登录。
7. 能看到风险资料、股票会话和完整分析，即表示永久保存成功。

## 以后自己或别人怎么使用

- 仍然打开同一个 `.streamlit.app` 网址。
- 每个人都用自己的邮箱注册；不同账号的数据互相不可见。
- 在另一台设备上，只需登录同一邮箱账号，不必重新部署程序。

## 常见问题

### 页面显示“尚未配置 SUPABASE_URL”

说明第四步的 Secrets 没有保存成功，或变量名拼写错误。两个变量名必须完全一致。

### 页面显示表不存在或云端资料读取失败

重新执行第二步的 `一键建表_V6.0.sql`，确认 SQL Editor 下方显示成功。

### GitHub 更新后网页仍是旧版

在 Streamlit Community Cloud 中点击 `⋮` → **Reboot app**，等待重新安装依赖。

### 收到的邮件仍然是确认链接

说明 Supabase 的 **Confirm signup** 邮件模板还没有改成 `{{ .Token }}`，请重新完成验证码邮件模板设置。

### 显示 email rate limit exceeded

这是 Supabase 邮件发送限额，不是验证码代码错误。不要连续点击，等待限额恢复；多人使用时需配置自定义 SMTP。

### 能否删除电脑上的压缩包

部署和跨设备验证成功后可以删除。建议保留一份 V6.4 压缩包作为本地备份。
