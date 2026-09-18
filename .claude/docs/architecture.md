# 架构

当前方法和流水线以 `doc/method.md` 为准。配置键以 `doc/config.md` 为准。

RP / MRP / Transition 都从 `representation/recurrence_plot.py` 的 `build_representation` 进入。三集数据与训练在 `scripts/experiments/`。旧两池入口是 `main_pipeline.py`，不是当前主线。
