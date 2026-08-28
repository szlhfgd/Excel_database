# 01: Dash 脚手架与布局骨架

**What to build:** 把 `app.py` 从 Streamlit 改写为可运行的 Dash 单页应用骨架：新建 `requirements.txt`（含 `dash`、`dash-bootstrap-components`，移除 `streamlit`），用 `dbc.Container/Row/Col` 搭出两栏布局（左侧控制栏占位、右侧主区占位），`app.run(debug=True, port=8050)` 入口；并加 `tests/test_app.py` 冒烟测试断言 `app` 是 `Dash` 实例且布局含关键 `id`。用户能启动页面并看到两栏外壳。

**Blocked by:** None (can start immediately)

**Status:** superseded

> 用户后续改回 Streamlit（见 `02-streamlit-rebuild.md`）。本脚手架的 Dash 实现已被取代。

- [x] `requirements.txt` 列出 dash / dash-bootstrap-components 及既有后端依赖，不含 streamlit
- [x] `app.py` 构建 `dash.Dash` 实例并注册 `app.run(debug=True, port=8050)` 入口
- [x] 两栏布局渲染：左侧控制栏占位 + 右侧主区占位，UI 文案为中文
- [x] `tests/test_app.py` 冒烟测试通过：导入 `app`，断言其为 `Dash` 实例且布局包含上传/表多选/模式切换/结果表/下载等关键 `id`
- [x] `python app.py` 能在本机 8050 启动且不报导入错误
