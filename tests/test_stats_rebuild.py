# -*- coding: utf-8 -*-
"""gpu_panel_v4 _stats_rebuild 逻辑离线测试：模拟 3 天数据 + 一次 llama-server 重启"""
import json
import time

import os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpu_panel.py")
src = open(SRC, encoding="utf-8").read()
# 去掉隐藏控制台 / stdout 重定向块，避免测试时把终端藏掉
src = src.replace('''if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except Exception:
        pass''', "")
src = src.replace("if sys.stdout is None:", "if False:")

ns = {"__name__": "gpu_panel_test",
      "__file__": SRC,
      "SCRIPT_DIR": "."}  # 会被被测模块用 __file__ 重新赋值，这里仅占位
exec(compile(src, SRC, "exec"), ns)

# ---- 构造 3 天模拟数据 ----
STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_stats.jsonl")
now = time.time()
t0 = now - 3 * 86400
rows = []
gt = 1000.0        # llama-server 已跑过一段的计数
pt = 5000.0
ps = 100.0
dt_ = 8000.0
at_ = 5600.0
ct_ = 20000.0
t = t0
reset_done = False
while t < now:
    busy = 0.0
    # 每天白天 10:00-22:00 有一半时间在推理
    import datetime as _dt
    hh = _dt.datetime.fromtimestamp(t).hour
    if 10 <= hh < 22:
        busy = 30.0  # 60s 窗口里忙 30s
        gt += 40 * 30      # 40 t/s * 30s
        pt += 1500
        ps += 30
        dt_ += 50 * 30     # MTP: 每 30s 消耗 1500 个 draft token
        at_ += 50 * 30 * 0.7
        ct_ += 3000        # 缓存命中：每拍 3000
    # 第 2 天 12:00 模拟 llama-server 重启：计数器清零重来
    if not reset_done and t > t0 + 1.5 * 86400:
        gt, pt, ps, dt_, at_, ct_ = 5.0, 10.0, 0.5, 5.0, 3.0, 5.0
        reset_done = True
    rows.append({"t": round(t, 1),
                 "ts": _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                 "bs": busy, "gs": (20.0 if 10 <= hh < 22 else 0.0),
                 "u": (55.0 if 10 <= hh < 22 else 0.0),
                 "temp": (60.0 if 10 <= hh < 22 else 35.0),
                 "gt": round(gt), "pt": round(pt),
                 "ps": round(ps, 3), "prs": round(ps * 3, 3),
                 "dt": round(dt_), "at": round(at_), "ct": round(ct_)})
    t += 60

with open(STATS, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

ns["STATS_LOG"] = STATS
ns["_stats_rebuild"]()
cache = ns["_stats_cache"]

w = cache["windows"]
print("windows:")
for k in ("24h", "7d", "30d", "1y"):
    print(" ", k, json.dumps(w[k], ensure_ascii=False))
print("data_from:", cache["data_from"], "rows:", cache["rows"])
print("hourly[:3]:", json.dumps(cache["buckets"]["hourly"][:3], ensure_ascii=False))
print("daily[-3:]:", json.dumps(cache["buckets"]["daily"][-3:], ensure_ascii=False))
print("weekly n =", len(cache["buckets"]["weekly"]),
      "last:", json.dumps(cache["buckets"]["weekly"][-1], ensure_ascii=False))

# ---- 断言 ----
# 重启前每天: 12h * 60 拍 * 30s 忙 = 21600s 忙; token 40*30*720拍 = 864000 tok/天
# 重启后 ~1.5 天: token ≈ 40*30*720 = 864000 (重启那段差分被丢弃)
exp_daily_tok = 40 * 30 * 720
gt24 = w["24h"]["tok_gen"]
assert 0.8 * exp_daily_tok < gt24 < 1.2 * exp_daily_tok, f"24h token 异常: {gt24}"
assert w["7d"]["tok_gen"] < 3 * exp_daily_tok * 1.1, f"7d token 应丢弃重启段差分: {w['7d']['tok_gen']}"
bs7 = w["7d"]["busy_s"]
assert 2 * 21600 * 0.9 < bs7 < 3 * 21600 * 1.1, f"7d 忙碌秒异常: {bs7}"
assert w["24h"]["avg_tps"] and 30 < w["24h"]["avg_tps"] < 50, f"avg_tps 异常: {w['24h']['avg_tps']}"
assert w["24h"]["temp_max"] == 60.0 and w["24h"]["gpu_util_avg"] > 20
assert len(cache["buckets"]["daily"]) == 30 and len(cache["buckets"]["hourly"]) == 24
assert abs(w["24h"]["spec_acc"] - 70.0) < 1, f"spec_acc 异常: {w['24h']['spec_acc']}"
# 缓存命中：12h * 720 拍 * 3000/拍 → 但每拍只有忙的时候才加（在 if 里），与 pt 同窗口
exp_cached = 3000 * 720
assert 0.8 * exp_cached < w["24h"]["tok_cached"] < 1.2 * exp_cached, \
    f"tok_cached 异常: {w['24h']['tok_cached']}"
# 重启点附近的小时桶不应出现爆炸值
hb = [b["tok"] for b in cache["buckets"]["hourly"]]
assert all(x <= exp_daily_tok / 12 for x in hb), f"小时桶有异常值: {max(hb)}"
print("ALL_ASSERTIONS_PASSED")
