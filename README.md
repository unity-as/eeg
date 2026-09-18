# EEG — 状态转移网络 + CNN

OpenNeuro ds002680，个体内 Go-nogo 四分类。主线是老师的状态转移矩阵 + CNN。RP 三集结果只作 baseline。

## 文档

只维护这三份，不要把同一件事再写到别处：

- `doc/method.md`：协议、当前表示、命令、还没做的事
- `doc/results.md`：全部数字
- `doc/config.md`：配置键

进度在 `.claude/records/status.md`。老师原话在 `doc/teacher_improvement_directions.md`。

## 运行

```bash
pip install -r requirements.txt

python scripts/experiments/build_correct_only_dataset.py --config config/eeg_ws_transition_3way.yaml
python scripts/experiments/train_3way.py --config config/eeg_ws_transition_3way.yaml
```

test 默认封闭。细节见 `doc/method.md`。
