# Streamlit UI/UX 优化说明

本次优化聚焦“concise 的办公软件风格”，参考飞书、钉钉和企业级中后台设计体系，不改动会议创建、导入、API 调用、数据库 CRUD、Agent 触发等业务逻辑。

## 主要改动

1. 新增 `ui_style.py`：集中管理主题 token、CSS、页面头部、侧边导航、流程 stepper。
2. 新增 `.streamlit/config.toml`：设置浅色企业后台主题。
3. 首页改为工作台卡片布局：减少长说明文本，改为模块入口 + 简短说明。
4. 统一所有页面的页头：使用紧凑 hero header，保持标题、说明、业务场景一致。
5. 统一侧边栏视觉：增加“导航 / 状态 / 控制”分区标签，弱化装饰性 emoji，提升扫描效率。
6. 会前准备流程改为垂直 stepper：当前步骤、已完成步骤、未开始步骤更清晰。
7. 表格、表单、按钮、上传区、指标卡、tabs、expander 做轻量卡片化和圆角统一。
8. 修复页面文件名编码：将 `#Uxxxx` 文件名恢复为中文页面名，使 Streamlit 页面导航更可读，也让原有 `st.page_link` 路径可正常匹配。

## 设计原则

- 信息密度：保留办公软件常见的高信息密度，但用边框、浅背景和留白分组降低视觉噪声。
- 可预测布局：基于 4px/8px 间距节奏，卡片、按钮和输入框尺寸更统一。
- 主操作突出：primary button 使用统一蓝色，普通操作保持白底描边。
- 业务不中断：本次未修改存储层、API endpoint、session_state key、数据库表操作和 Agent 调用逻辑。

## 运行方式

```bash
streamlit run app_streamlit/Home.py --server.port 8501
```
