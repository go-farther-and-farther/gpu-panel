# gpu-panel

单文件 GPU / 系统 / llama.cpp 监控面板（V4）。

> 🚧 README 编写中 —— 功能清单与截图待补（等面板积累几天真实数据）。
> 设计目标、待办事项见 `docs/` 或联系作者。

## 快速开始（预览）

```bash
pip install psutil
py gpu_panel.py            # 监听 0.0.0.0:8081
```

浏览器打开 `http://127.0.0.1:8081/`。

- GPU 指标来自 `nvidia-smi`（NVIDIA 显卡）
- 推理指标来自 llama.cpp server（`/slots`、`/metrics`），默认地址 `http://127.0.0.1:8080`，
  可用环境变量 `LLM_URL` 覆盖
