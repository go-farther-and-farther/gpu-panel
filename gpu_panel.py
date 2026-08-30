# -*- coding: utf-8 -*-
"""
gpu_panel_v4 - GPU / 系统 / llama.cpp 监控面板 V4
- 每 2 秒采样：NVIDIA GPU（支持多卡）、内存、CPU、磁盘、Top 进程、llama.cpp slots
- 30 分钟内存历史（900 条 @2s）+ CSV 日志（默认每 10s 落盘一行，5MB 滚动）
- Web 面板：继承 V3 修复（趋势图 / sparkline / 断线保护 / 4K 居中 / 移动端）
- 升级 llama.cpp 推理区：总输出速度 / 总平均速度 / 总 Prompt 速度 /
  观察生成 token / 任务时长 / 剩余预估 / 30 分钟总速度 sparkline
- 监听 0.0.0.0:8081（V1 8321 · V2 8322 · V3 8323，互不影响）
- 采样分层：llama.cpp /slots 每 2s（token 速度要快），nvidia-smi + psutil 每 10s（重，放慢）
- 长期工作统计：每 60s 快照 token 计数器/忙闲增量 → stats_log.jsonl（保留 1 年），
  聚合出 24h / 7天 / 30天 / 1年 的生成 token、Prompt token、推理忙碌时长等（/api/stats）

用法:  py gpu_panel_v4.py [port]
环境变量:
  LLM_URL          llama-server 地址 (默认 http://127.0.0.1:8080)
  SAMPLE_INTERVAL  采样周期秒 (默认 2)
  MAX_HISTORY      内存历史条数 (默认 900 ≈ 30 分钟 @2s)
  CSV_INTERVAL     CSV 落盘间隔秒 (默认 10；设 0 或负数 = 完全关闭 CSV 日志)
  PANEL_TOKEN      设置后 /api/* 需要 ?token=xxx 或 Authorization: Bearer xxx
                    （页面 URL 带 ?token=xxx 时前端自动带上）
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import psutil

# ---- Windows: 隐藏控制台窗口 ----
# 双击 .py 或 `py gpu_panel_v4.py` 启动时，把黑色命令行窗口隐藏（进程继续后台运行）。
# 注意：若是在 cmd 窗口里手动运行本脚本，被隐藏的会是那个 cmd 窗口本身。
# 想恢复显示窗口：删除下面这个 if 块即可。
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except Exception:
        pass

# pythonw.exe（.pyw）下没有控制台，sys.stdout/stderr 为 None，print() 会抛异常，
# 这里把它们重定向到空设备，保证两种启动方式都能正常运行。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8081
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080")
try:
    SAMPLE_INTERVAL = max(1.0, float(os.environ.get("SAMPLE_INTERVAL", "2")))
except ValueError:
    SAMPLE_INTERVAL = 2.0
try:
    MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "900"))
except ValueError:
    MAX_HISTORY = 900
MAX_HISTORY = max(60, MAX_HISTORY)
PANEL_TOKEN = os.environ.get("PANEL_TOKEN", "")
CSV_LOG = os.path.join(SCRIPT_DIR, "gpu_log_v4.csv")
CSV_MAX_BYTES = 5 * 1024 * 1024
try:
    CSV_INTERVAL = float(os.environ.get("CSV_INTERVAL", "10"))  # 秒；<=0 表示关闭 CSV 日志
except ValueError:
    CSV_INTERVAL = 10.0
try:
    SYS_INTERVAL = max(2.0, float(os.environ.get("SYS_INTERVAL", "10")))  # GPU/系统采样周期秒
except ValueError:
    SYS_INTERVAL = 10.0
STATS_LOG = os.path.join(SCRIPT_DIR, "stats_log.jsonl")
STATS_INTERVAL = 60.0          # 长期统计采样周期秒
STATS_RETENTION_DAYS = 366     # 统计数据保留天数
START_TIME = time.time()

RECENT_WINDOW_S = 3.0  # “3s 速度”滑动窗口，对齐 llama-server 表格
RECENT_KEEP_S = 10.0   # 每个 slot 保留的 (t, n_decoded) 样本时长上限

_history = deque(maxlen=MAX_HISTORY)
_history_lock = threading.Lock()
_csv_init = {"done": False}
_last_csv_t = {"t": 0.0}

# llama.cpp 速率计算：
#   _llama_prev[sid]  = 上一次采样的 (n_decoded, n_prompt_processed, task_id, t)
#   _llama_track[sid] = 当前任务 {task, t0, d0, samples:[(t, n_decoded)...]}
_llama_prev = {}
_llama_track = {}
_sess_decoded = [0]  # 面板启动以来观察到的生成 token 累计（按任务差分累加）

# 快慢双采样共享的"当前快照"：快循环(llama 2s)每拍刷新 ts 并 append 历史，
# 慢循环(GPU/系统 10s)整块替换 gpu / ram / cpu 等键，浅拷贝安全
_snap_lock = threading.Lock()
_current = {}
_busy_s = [0.0]        # 本统计窗口内推理忙碌累计秒（fast 循环累加，stats 线程取走清零）
_gpu_active_s = [0.0]  # 本统计窗口内 GPU 活跃累计秒 util>10%（slow 循环累加）
_stats_lock = threading.Lock()
_stats_cache = {"ready": False}


def _n(v):
    """容错转 float，失败返回 None"""
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def sample_gpu():
    """nvidia-smi，支持多卡。返回 {"gpus": [...], "error": None|msg}"""
    # pythonw（.pyw）下父进程没有控制台，nvidia-smi 每次都会闪出黑色窗口；
    # 加 CREATE_NO_WINDOW 让子进程不创建新控制台。
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=flags).stdout
        gpus = []
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            gpus.append({
                "index": int(_n(f[0]) or 0),
                "name": f[1],
                "util": _n(f[2]),
                "mem_used": _n(f[3]),
                "mem_total": _n(f[4]),
                "temp": _n(f[5]),
                "power": _n(f[6]),
                "fan": _n(f[7]) if len(f) > 7 else None,
            })
        if not gpus:
            return {"gpus": [], "error": "no output"}
        return {"gpus": gpus, "error": None}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


def sample_system():
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            mi = p.info["memory_info"]
            procs.append({"pid": p.info["pid"], "name": p.info["name"] or "?",
                          "mem_mb": round(mi.rss / 1048576, 1)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, TypeError):
            pass
    procs.sort(key=lambda x: -x["mem_mb"])
    disk = {}
    try:
        root = "C:\\" if os.name == "nt" else "/"
        du = shutil.disk_usage(root)
        if du.total > 0:
            disk = {
                "label": root[:-1],
                "total_gb": round(du.total / 1073741824, 1),
                "free_gb": round(du.free / 1073741824, 1),
                "pct": round(du.used / du.total * 100, 1),
            }
    except Exception:
        pass
    return {
        "ram_total_mb": vm.total / 1048576,
        "ram_used_mb": vm.used / 1048576,
        "ram_pct": vm.percent,
        "cpu": cpu,
        "disk": disk,
        "top_procs": procs[:10],
    }


def _llm_get(path, timeout=3):
    with urllib.request.urlopen(LLM_URL.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _metrics_get():
    """拉取 llama-server Prometheus 文本指标，返回 {name: float}"""
    with urllib.request.urlopen(LLM_URL.rstrip("/") + "/metrics", timeout=3) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def sample_llama():
    """查询 llama-server /slots，差分计算每个 slot 的 3s 窗口速度 / 任务平均速度 / prompt 速度"""
    base = {"url": LLM_URL, "online": False, "error": None, "slots": [], "busy": 0, "total": 0}
    try:
        slots_raw = _llm_get("/slots")
        if not isinstance(slots_raw, list):
            raise ValueError("bad /slots response")
    except Exception as e:
        base["error"] = str(e)
        return base

    now = time.time()
    slots = []
    busy = 0
    for raw in slots_raw:
        if not isinstance(raw, dict):
            continue
        sid = raw.get("id")
        task = raw.get("id_task")
        ctx = raw.get("n_ctx", 0) or 0
        n_prompt = raw.get("n_prompt_tokens", 0) or 0
        n_prompt_proc = raw.get("n_prompt_tokens_processed", 0) or 0
        nt = (raw.get("next_token") or [{}])[0]
        if not isinstance(nt, dict):
            nt = {}
        n_decoded = nt.get("n_decoded", 0) or 0
        n_remain = nt.get("n_remain")
        processing = bool(raw.get("is_processing"))
        if processing:
            busy += 1

        # --- 记录上次采样 + 观察生成 token 累计 ---
        prev = _llama_prev.get(sid)
        if prev and task == prev.get("task"):
            d_dec = n_decoded - prev.get("n_decoded", 0)
            if d_dec > 0:
                _sess_decoded[0] += d_dec
        _llama_prev[sid] = {"t": now, "n_decoded": n_decoded,
                            "n_prompt_processed": n_prompt_proc, "task": task}

        # --- 任务追踪：任务切换（task id 变化）时重置 ---
        tr = _llama_track.get(sid)
        if tr is None or tr.get("task") != task:
            tr = {"task": task, "t0": now, "d0": n_decoded, "p0": n_prompt_proc,
                  "samples": deque(maxlen=64), "psamples": deque(maxlen=64)}
            _llama_track[sid] = tr
        tr["samples"].append((now, n_decoded))
        tr["psamples"].append((now, n_prompt_proc))
        cutoff = now - RECENT_KEEP_S
        while tr["samples"] and tr["samples"][0][0] < cutoff:
            tr["samples"].popleft()
        while tr["psamples"] and tr["psamples"][0][0] < cutoff:
            tr["psamples"].popleft()
        task_elapsed = round(now - tr["t0"], 1) if processing else None

        # --- Prompt (prefill) 速度：10s 滑动窗口 ---
        # n_prompt_tokens_processed 按 batch 跳变（chunked prefill），相邻差分
        # 会在 0 和 batch 速率之间横跳；10s 窗口跨多个 batch，毛刺被抹平
        prompt_tps = None
        target = now - RECENT_KEEP_S
        ref_t, ref_p = tr["t0"], tr["p0"]
        for (st, sp) in tr["psamples"]:
            if st <= target:
                ref_t, ref_p = st, sp
            else:
                break
        d_pp = n_prompt_proc - ref_p
        if d_pp > 0:
            prompt_tps = round(d_pp / max(now - ref_t, 1e-6), 1)

        gen_recent = gen_avg = None
        if processing and n_decoded > 0:
            # 任务平均速度：从任务开始累计
            dt_avg = max(now - tr["t0"], 1e-6)
            d_avg = n_decoded - tr["d0"]
            if d_avg > 0:
                gen_avg = round(d_avg / dt_avg, 1)
            # 3s 窗口速度：找最接近 now-3s 的样本做基准
            target = now - RECENT_WINDOW_S
            ref_t, ref_d = tr["t0"], tr["d0"]
            for (st, sd) in tr["samples"]:
                if st <= target:
                    ref_t, ref_d = st, sd
                else:
                    break
            dt_r = max(now - ref_t, 1e-6)
            d_r = n_decoded - ref_d
            if d_r > 0:
                gen_recent = round(d_r / dt_r, 1)

        remain_est = None
        if processing and n_remain and n_remain > 0:
            sp = gen_recent if gen_recent else gen_avg
            if sp:
                remain_est = round(n_remain / sp, 1)

        slots.append({
            "id": sid,
            "processing": processing,
            "task": task,
            "ctx": ctx,
            "ctx_used": n_prompt,
            "ctx_pct": round(n_prompt / ctx * 100, 1) if ctx else 0,
            "n_decoded": n_decoded,
            "gen_recent_tps": gen_recent,
            "gen_avg_tps": gen_avg,
            "prompt_tps": prompt_tps,
            "task_elapsed": task_elapsed,
            "remain_est": remain_est,
            "has_next": nt.get("has_next_token"),
            "n_remain": n_remain,
            "speculative": raw.get("speculative"),
        })

    tot_recent = round(sum(sl["gen_recent_tps"] or 0 for sl in slots), 1)
    tot_avg = round(sum(sl["gen_avg_tps"] or 0 for sl in slots), 1)
    tot_prompt = round(sum(sl["prompt_tps"] or 0 for sl in slots), 1)
    if busy > 0:
        _busy_s[0] += SAMPLE_INTERVAL  # 本拍有 slot 在生成 → 计入推理忙碌时长
    base.update({"online": True, "slots": slots, "busy": busy, "total": len(slots),
                 "tot_recent_tps": tot_recent,
                 "tot_avg_tps": tot_avg,
                 "tot_prompt_tps": tot_prompt,
                 "sess_decoded": _sess_decoded[0]})
    return base


def _write_csv(s):
    g = (s.get("gpu", {}).get("gpus") or [{}])[0]
    top = (s.get("top_procs") or [{}])[0]
    L = s.get("llama") or {}
    tps = max([sl.get("gen_recent_tps") or 0 for sl in L.get("slots", [])], default=0)
    row = [
        s["ts"],
        g.get("util", ""), g.get("mem_used", ""), g.get("mem_total", ""),
        g.get("temp", ""), g.get("power", ""),
        round(s.get("ram_used_mb", 0)), round(s.get("ram_pct", 0), 1), s.get("cpu", ""),
        top.get("name", ""), top.get("mem_mb", ""),
        ("llama_offline" if L.get("error")
         else "%d/%d" % (L.get("busy", 0), L.get("total", 0))),
        tps,
    ]
    try:
        rotated = False
        if os.path.exists(CSV_LOG) and os.path.getsize(CSV_LOG) > CSV_MAX_BYTES:
            try:
                os.replace(CSV_LOG, CSV_LOG[:-4] + "_old.csv")
                rotated = True
            except OSError:
                pass
        new = rotated or not _csv_init["done"]
        with open(CSV_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "gpu0_util%", "gpu0_vram_used_MB", "gpu0_vram_total_MB",
                            "gpu0_temp_C", "gpu0_power_W", "ram_used_MB", "ram_pct%", "cpu_pct%",
                            "top_proc", "top_proc_mem_MB", "llama_busy", "llama_gen3s_tps"])
            w.writerow(row)
        _csv_init["done"] = True
    except Exception:
        pass


def sampler_loop_fast():
    """快循环（默认 2s）：只采 llama.cpp /slots，刷新时间戳并 append 历史/CSV"""
    while True:
        try:
            llama = sample_llama()
            now = time.time()
            with _snap_lock:
                _current["llama"] = llama
                _current["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _current["t"] = now
                _current["uptime_s"] = int(now - START_TIME)
                s = dict(_current)
            with _history_lock:
                _history.append(s)
            # CSV 按 CSV_INTERVAL 间隔落盘（默认 10s），避免日志增长过快；
            # CSV_INTERVAL <= 0 时完全关闭 CSV 日志。
            if CSV_INTERVAL > 0 and now - _last_csv_t["t"] >= CSV_INTERVAL:
                _write_csv(s)
                _last_csv_t["t"] = now
        except Exception:
            time.sleep(1)
        time.sleep(SAMPLE_INTERVAL)


def sampler_loop_slow():
    """慢循环（默认 10s）：nvidia-smi（每次拉起子进程，开销大）+ psutil"""
    psutil.cpu_percent(interval=None)  # 初始化基线
    while True:
        try:
            gpu = sample_gpu()
            sysd = sample_system()
            with _snap_lock:
                _current["gpu"] = gpu
                _current.update(sysd)
            gpus = gpu.get("gpus") or []
            if gpus and (gpus[0].get("util") or 0) > 10:
                _gpu_active_s[0] += SYS_INTERVAL
        except Exception:
            pass
        time.sleep(SYS_INTERVAL)


# ---------------- 长期工作统计（stats_log.jsonl） ----------------
# 每 60s 一行快照：token 计数器存原始累计值（聚合时差分，自动处理 llama-server 重置），
# 推理忙碌/GPU 活跃存本窗口增量秒（面板重启不影响口径）。

def _stats_record():
    now = time.time()
    with _snap_lock:
        gpus = (_current.get("gpu") or {}).get("gpus") or []
        busy_s = _busy_s[0]
        gpu_s = _gpu_active_s[0]
    _busy_s[0] = 0.0
    _gpu_active_s[0] = 0.0
    row = {
        "t": round(now, 1),
        "ts": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
        "bs": round(busy_s, 1),
        "gs": round(gpu_s, 1),
        "u": gpus[0].get("util") if gpus else None,
        "temp": gpus[0].get("temp") if gpus else None,
    }
    try:
        m = _metrics_get()
        row["gt"] = m.get("llamacpp:tokens_predicted_total")
        row["pt"] = m.get("llamacpp:prompt_tokens_total")
        row["ps"] = round(m.get("llamacpp:tokens_predicted_seconds_total") or 0, 3)
        row["prs"] = round(m.get("llamacpp:prompt_seconds_total") or 0, 3)
        # 投机解码（MTP）：draft/accepted 累计计数器，聚合时算接受率
        row["dt"] = m.get("llamacpp:spec_decode_num_draft_tokens_total")
        row["at"] = m.get("llamacpp:spec_decode_num_accepted_tokens_total")
        # 缓存命中 token（与实际计算的 prompt token 是两个独立计数器，相加 = 输入总量）
        row["ct"] = m.get("llamacpp:prompt_tokens_cached_total")
    except Exception:
        pass  # llama 离线：该行只记忙闲/系统指标
    try:
        with open(STATS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _stats_prune():
    """删除超过保留期（默认 1 年）的旧行"""
    if not os.path.exists(STATS_LOG):
        return
    cutoff = time.time() - STATS_RETENTION_DAYS * 86400
    kept = []
    try:
        with open(STATS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("t", 0) >= cutoff:
                    kept.append(line if line.endswith("\n") else line + "\n")
        tmp = STATS_LOG + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, STATS_LOG)
    except Exception:
        pass


def _stats_rebuild():
    """全量扫描 stats_log，聚合出 24h/7d/30d/1y 窗口与 小时/天/周 分桶，缓存供 /api/stats"""
    rows = []
    try:
        with open(STATS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    rows.sort(key=lambda r: r.get("t", 0))
    now = time.time()
    WINS = [("24h", 86400), ("7d", 7 * 86400), ("30d", 30 * 86400), ("1y", 365 * 86400)]

    def _new_acc():
        return {"gt": 0.0, "pt": 0.0, "ct": 0.0, "ps": 0.0, "dt": 0.0, "at": 0.0,
                "bs": 0.0, "gs": 0.0, "us": [], "temps": []}

    accs = {k: _new_acc() for k, _s in WINS}
    hacc, dacc, wacc = {}, {}, {}  # 分桶：小时 / 天 / 周

    def _bucket_for(t):
        out = []
        if t >= now - 86400:
            out.append(hacc.setdefault(int(t // 3600), _new_acc()))
        if t >= now - 30 * 86400:
            day = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
            out.append(dacc.setdefault(day, _new_acc()))
        dt_ = datetime.fromtimestamp(t)
        ws = (dt_ - timedelta(days=dt_.weekday())).strftime("%Y-%m-%d")  # 周一为界
        out.append(wacc.setdefault(ws, _new_acc()))
        return out

    prev = None
    for r in rows:
        t = r.get("t", 0)
        if t < now - 365 * 86400:
            continue
        for k, s in WINS:
            if t >= now - s:
                a = accs[k]
                a["bs"] += r.get("bs") or 0
                a["gs"] += r.get("gs") or 0
                if r.get("u") is not None:
                    a["us"].append(r["u"])
                if r.get("temp") is not None:
                    a["temps"].append(r["temp"])
        for b in _bucket_for(t):
            b["bs"] += r.get("bs") or 0
            b["gs"] += r.get("gs") or 0
        gt = r.get("gt")
        if gt is not None:
            if prev is not None:
                # 计数器变小 = llama-server 重启过 → 这一段差分丢弃
                d_gt = max(0.0, gt - prev.get("gt", 0))
                d_pt = max(0.0, (r.get("pt") or 0) - (prev.get("pt") or 0))
                d_ps = max(0.0, (r.get("ps") or 0) - (prev.get("ps") or 0))
                d_dt = max(0.0, (r.get("dt") or 0) - (prev.get("dt") or 0))
                d_at = max(0.0, (r.get("at") or 0) - (prev.get("at") or 0))
                d_ct = max(0.0, (r.get("ct") or 0) - (prev.get("ct") or 0))
                for k, s in WINS:
                    if t >= now - s:
                        a = accs[k]
                        a["gt"] += d_gt
                        a["pt"] += d_pt
                        a["ct"] += d_ct
                        a["ps"] += d_ps
                        a["dt"] += d_dt
                        a["at"] += d_at
                for b in _bucket_for(t):
                    b["gt"] += d_gt
                    b["pt"] += d_pt
                    b["ct"] += d_ct
                    b["dt"] += d_dt
                    b["at"] += d_at
            prev = r

    def _finish(a):
        return {
            "tok_gen": round(a["gt"]), "tok_prompt": round(a["pt"]),
            "tok_cached": round(a["ct"]),
            "pred_s": round(a["ps"], 1),
            "busy_s": round(a["bs"], 1), "gpu_active_s": round(a["gs"], 1),
            "avg_tps": round(a["gt"] / a["ps"], 1) if a["ps"] > 0 else None,
            "spec_acc": round(a["at"] / a["dt"] * 100, 1) if a["dt"] > 0 else None,
            "gpu_util_avg": round(sum(a["us"]) / len(a["us"]), 1) if a["us"] else None,
            "temp_max": max(a["temps"]) if a["temps"] else None,
        }

    # 稠密分桶：没数据的时间格补 0，前端柱状图才有连续时间轴
    def _bucket_val(acc, key):
        a = acc.get(key) or _new_acc()
        return round(a["gt"]), round(a["bs"], 1)

    hourly = []
    for i in range(23, -1, -1):
        k = int((now - i * 3600) // 3600)
        tok, bs = _bucket_val(hacc, k)
        hourly.append({"label": datetime.fromtimestamp(k * 3600).strftime("%d日 %H:00"),
                       "tok": tok, "busy_s": bs})
    today = datetime.fromtimestamp(now).date()
    daily = []
    for i in range(29, -1, -1):
        day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        tok, bs = _bucket_val(dacc, day)
        daily.append({"label": day[5:], "tok": tok, "busy_s": bs})
    week0 = today - timedelta(days=today.weekday())
    weekly = []
    for i in range(52, -1, -1):
        ws = (week0 - timedelta(days=7 * i)).strftime("%Y-%m-%d")
        tok, bs = _bucket_val(wacc, ws)
        weekly.append({"label": ws[5:], "tok": tok, "busy_s": bs})

    cache = {
        "ready": True,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_from": rows[0].get("ts") if rows else None,
        "rows": len(rows),
        "windows": {k: _finish(accs[k]) for k, _s in WINS},
        "buckets": {"hourly": hourly, "daily": daily, "weekly": weekly},
    }
    with _stats_lock:
        _stats_cache.clear()
        _stats_cache.update(cache)


def stats_loop():
    _stats_prune()
    last_day = datetime.now().strftime("%Y-%m-%d")
    while True:
        try:
            _stats_record()
            _stats_rebuild()
        except Exception:
            pass
        if datetime.now().strftime("%Y-%m-%d") != last_day:  # 每天 prune 一次
            last_day = datetime.now().strftime("%Y-%m-%d")
            _stats_prune()
        time.sleep(STATS_INTERVAL)


PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0d1220">
<title>GPU / 系统监控面板 V4</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='3' y='3' width='18' height='18' rx='4' fill='%23ff7a59'/%3E%3Crect x='7.5' y='7.5' width='9' height='9' rx='2' fill='%230d1220'/%3E%3C/svg%3E">
<style>
:root{
  --bg:#0d1220; --card:#151c2e; --line:rgba(148,163,184,.10);
  --fg:#e8eefb; --dim:#8593ad; --dim2:#5b6884;
  --gpu:#ff7a59; --ram:#4da3ff; --cpu:#7ee081; --disk:#c792ea;
  --temp:#ffd166; --warn:#ffd166; --err:#ff6b6b; --ok:#7ee081;
  --r:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(1100px 480px at 85% -10%, rgba(77,163,255,.07), transparent 60%),
    radial-gradient(900px 420px at -10% 110%, rgba(255,122,89,.05), transparent 60%),
    var(--bg);
  color:var(--fg);
  font:14px/1.55 "Segoe UI Variable Text","Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  padding:22px clamp(14px,3vw,40px) 30px;
  max-width:1560px;margin:0 auto;min-height:100vh;
}
h1{font-size:20px;font-weight:650;letter-spacing:.2px}
.ver{font-size:10.5px;color:#7db8ff;border:1px solid rgba(77,163,255,.35);
     padding:1px 8px;border-radius:999px;vertical-align:3px;margin-left:6px;letter-spacing:.5px}
.sub{color:var(--dim);font-size:12.5px;margin-top:4px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.hdr-right{display:flex;gap:8px;flex-wrap:wrap;padding-top:3px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:4px 11px;
      border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.03);white-space:nowrap}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--dim2)}
.pill.ok .dot{background:var(--ok);box-shadow:0 0 8px rgba(126,224,129,.8)}
.pill.off .dot{background:var(--err)}
.pill.dim{color:var(--dim)}
.banner{margin:14px 0 0;padding:10px 14px;border-radius:10px;font-size:13px;
        background:rgba(255,107,107,.10);border:1px solid rgba(255,107,107,.35);color:#ffb4b4}
.hidden{display:none!important}
/* ---- 指标卡片 ---- */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:18px 0 4px}
.card{position:relative;background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,0) 45%),var(--card);
      border:1px solid var(--line);border-radius:var(--r);padding:13px 16px 8px;overflow:hidden}
.card::before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--c,#334466)}
.card .label{color:var(--dim);font-size:12px;letter-spacing:.3px}
.card .value{font-size:clamp(21px,1.9vw,26px);font-weight:650;margin-top:6px;
             font-variant-numeric:tabular-nums;letter-spacing:.3px}
.card .value small{font-size:13px;color:var(--dim);font-weight:500}
.card .extra{color:var(--dim);font-size:12px;margin-top:2px;font-variant-numeric:tabular-nums}
.bar{height:6px;background:rgba(255,255,255,.06);border-radius:99px;margin-top:10px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:var(--c);width:0;transition:width .6s ease}
.spark{display:block;width:100%;height:36px;margin-top:8px}
/* ---- 面板 ---- */
.panel{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
       padding:16px 18px;margin:16px 0}
.panel h2{font-size:13px;color:var(--dim);font-weight:600;letter-spacing:.4px;margin-bottom:12px;
          display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.panel h2 .title{color:#b9c5dd;font-weight:600}
.scroller{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--dim2);font-size:11px;font-weight:600;letter-spacing:.7px;text-align:left;
   padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid rgba(148,163,184,.06);
   font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.02)}
.mono{font-family:"Cascadia Code",Consolas,monospace;font-size:12px;color:#b9c5dd}
.err{color:var(--err)}
.dim{color:var(--dim)}
/* 状态 */
.st{display:inline-flex;align-items:center;gap:6px;font-size:12.5px}
.st .dot{width:7px;height:7px;border-radius:50%}
.st.busy{color:var(--warn)}
.st.busy .dot{background:var(--warn);animation:pulse 1.2s ease-in-out infinite}
.st.idle{color:var(--dim)}
.st.idle .dot{background:var(--dim2)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.tag{background:rgba(77,163,255,.15);color:#7db8ff;border-radius:5px;
     padding:1px 7px;font-size:11px;margin-left:8px;white-space:nowrap}
/* 上下文进度条 */
.ctx{display:flex;align-items:center;gap:9px}
.ctxbar{width:96px;height:5px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;flex:none}
.ctxbar i{display:block;height:100%;border-radius:99px}
.ctx span{color:#b9c5dd;font-size:12.5px}
/* 进程内存条 */
.memcell{display:flex;align-items:center;gap:10px}
.membar{width:110px;height:5px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;flex:none}
.membar i{display:block;height:100%;border-radius:99px;background:var(--ram)}
/* 趋势图 */
#chart{width:100%;height:175px;display:block}
.legend{margin-left:auto;display:flex;gap:14px;align-items:center;font-weight:400}
.chip{display:inline-flex;align-items:center;gap:6px;color:var(--dim);font-size:12px}
#llama_chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px}
#llama_chips .chip{background:rgba(255,255,255,.03);border:1px solid var(--line);
  border-radius:999px;padding:3.5px 11px}
.chip i{width:9px;height:9px;border-radius:3px;display:inline-block}
.chip b{color:var(--fg);font-variant-numeric:tabular-nums}
.foot{color:var(--dim2);font-size:11.5px;text-align:center;margin-top:24px}
/* 工作统计 */
.stab{font-size:12px;color:var(--dim);border:1px solid var(--line);background:rgba(255,255,255,.03);
      border-radius:999px;padding:3px 12px;cursor:pointer;font-family:inherit}
.stab.on{color:#0d1220;background:#7db8ff;border-color:#7db8ff;font-weight:600}
#stats_cards{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:6px 0 12px}
#stats_cards .card{padding:10px 13px 6px}
#stats_cards .card .value{font-size:20px}
#stats_tip{position:fixed;display:none;z-index:50;pointer-events:none;
  background:rgba(13,18,32,.96);border:1px solid rgba(148,163,184,.25);border-radius:8px;
  padding:7px 11px;font-size:12px;line-height:1.7;color:var(--fg);
  box-shadow:0 4px 16px rgba(0,0,0,.45);white-space:nowrap;font-variant-numeric:tabular-nums}
/* ---- 移动端 ---- */
@media (max-width:720px){
  body{padding:16px 12px 24px}
  .grid{grid-template-columns:repeat(2,1fr);gap:10px}
  .card{padding:11px 12px 6px}
  .spark{height:28px}
  .panel{padding:13px 13px}
  #chart{height:140px}
  table.stack thead{display:none}
  table.stack tr{display:block;padding:9px 0;border-bottom:1px solid rgba(148,163,184,.08)}
  table.stack td{display:flex;justify-content:space-between;align-items:center;gap:10px;
                 padding:3.5px 0;border:none}
  table.stack td::before{content:attr(data-th);color:var(--dim);font-size:12px;flex:none}
  table.stack tr:last-child{border-bottom:none}
}
</style>
</head>
<body>
<header>
  <div>
    <h1>GPU / 系统监控面板<span class="ver">V4</span></h1>
    <div class="sub"><span id="gname">…</span> · 每 <span id="intv">…</span>s 采样 · 最近采样 <span id="ts">加载中…</span></div>
  </div>
  <div class="hdr-right">
    <span id="conn" class="pill"><span class="dot"></span>连接中…</span>
    <span id="uptime" class="pill dim">运行 —</span>
  </div>
</header>
<div id="banner" class="banner hidden"></div>

<section class="grid">
  <div class="card" id="card_gpu_util" style="--c:var(--gpu)">
    <div class="label">GPU 利用率</div>
    <div class="value" id="gpu_util">—</div>
    <div class="bar"><i id="gpu_util_bar"></i></div>
    <canvas class="spark" id="sp_util"></canvas>
  </div>
  <div class="card" id="card_gpu_mem" style="--c:var(--gpu)">
    <div class="label">显存</div>
    <div class="value" id="gpu_mem">—</div>
    <div class="extra" id="gpu_mem_extra"></div>
    <div class="bar"><i id="gpu_mem_bar"></i></div>
    <canvas class="spark" id="sp_vram"></canvas>
  </div>
  <div class="card" id="card_gpu_temp" style="--c:var(--temp)">
    <div class="label">GPU 温度 / 功耗</div>
    <div class="value" id="gpu_temp">—</div>
    <div class="extra" id="gpu_extra"></div>
    <canvas class="spark" id="sp_temp"></canvas>
  </div>
  <div class="card" style="--c:var(--ram)">
    <div class="label">内存 (RAM)</div>
    <div class="value" id="ram">—</div>
    <div class="extra" id="ram_extra"></div>
    <div class="bar"><i id="ram_bar"></i></div>
    <canvas class="spark" id="sp_ram"></canvas>
  </div>
  <div class="card" style="--c:var(--cpu)">
    <div class="label">CPU</div>
    <div class="value" id="cpu">—</div>
    <div class="bar"><i id="cpu_bar"></i></div>
    <canvas class="spark" id="sp_cpu"></canvas>
  </div>
  <div class="card" style="--c:var(--disk)">
    <div class="label" id="disk_label">磁盘</div>
    <div class="value" id="disk">—</div>
    <div class="extra" id="disk_extra"></div>
    <div class="bar"><i id="disk_bar"></i></div>
  </div>
</section>

<section class="panel hidden" id="multi_gpu_panel">
  <h2><span class="title">多 GPU</span></h2>
  <table>
    <thead><tr><th>型号</th><th>利用率</th><th>显存</th><th>温度</th><th>功耗</th><th>风扇</th></tr></thead>
    <tbody id="gpus_tbody"></tbody>
  </table>
</section>

<section class="panel">
  <h2><span class="title">llama.cpp 推理</span>
      <span id="llama_pill" class="pill"><span class="dot"></span>…</span>
      <span id="llama_meta" class="sub" style="margin:0"></span></h2>
  <div id="llama_chips"></div>
  <canvas class="spark" id="sp_llama"></canvas>
  <div class="scroller">
  <table>
    <thead><tr><th>Slot</th><th>状态</th><th>任务</th><th>3s 速度</th><th>平均速度</th><th>Prompt 速度</th>
    <th>任务时长</th><th>剩余预估</th><th>本轮已生成</th><th>上下文</th><th>投机</th></tr></thead>
    <tbody id="slots_tbody"></tbody>
  </table>
  </div>
</section>

<section class="panel">
  <h2><span class="title">趋势（最近 30 分钟）</span><span id="legend" class="legend"></span></h2>
  <canvas id="chart"></canvas>
</section>

<section class="panel">
  <h2><span class="title">工作统计</span>
      <span id="stats_tabs" style="display:flex;gap:6px;flex-wrap:wrap"></span>
      <span id="stats_metric" style="display:flex;gap:6px"></span>
      <span id="stats_meta" class="sub" style="margin:0"></span></h2>
  <div id="stats_cards" class="grid"></div>
  <canvas id="stats_chart" style="width:100%;height:150px;display:block"></canvas>
  <div id="stats_tip"></div>
</section>

<section class="panel">
  <h2><span class="title">内存占用 Top 进程</span></h2>
  <table class="stack">
    <thead><tr><th>PID</th><th>进程名</th><th>内存 MB</th></tr></thead>
    <tbody id="procs_tbody"></tbody>
  </table>
</section>

<section class="panel">
  <h2><span class="title">最近采样记录</span></h2>
  <div class="scroller">
  <table>
    <thead><tr><th>时间</th><th>GPU%</th><th>显存 MB</th><th>温度°C</th><th>RAM%</th><th>CPU%</th></tr></thead>
    <tbody id="log_tbody"></tbody>
  </table>
  </div>
</section>

<footer class="foot">gpu_panel_v4 · 内存历史 30 分钟 · 长期统计 1 年（stats_log.jsonl）· 数据源 nvidia-smi / psutil / llama.cpp /slots /metrics</footer>

<script>
"use strict";
var $ = function(id){ return document.getElementById(id); };
var esc = function(s){
  return String(s == null ? "" : s).replace(/[&<>"']/g, function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
  });
};
var fmt = function(v, d){
  d = (d == null) ? 1 : d;
  if (v == null || v === "" || isNaN(v)) return "—";
  return Number(v).toLocaleString("en-US", {minimumFractionDigits:0, maximumFractionDigits:d});
};
var clampPct = function(v){
  if (v == null || isNaN(v)) return null;
  v = Number(v);
  return Math.max(0, Math.min(100, v));
};
var COLORS = {gpu:"#ff7a59", ram:"#4da3ff", cpu:"#7ee081", disk:"#c792ea", temp:"#ffd166"};
var TOKEN = (function(){
  try { return new URLSearchParams(location.search).get("token") || ""; } catch(e){ return ""; }
})();
var Q = TOKEN ? "?token=" + encodeURIComponent(TOKEN) : "";
var histCache = [];
var lastData = null;
var connState = {ok:null, lastOk:null};

function setBar(id, p){
  var el = $(id);
  if (!el) return;
  p = clampPct(p);
  el.style.width = (p == null ? 0 : p) + "%";
}

function setConn(ok, err){
  var pill = $("conn"), banner = $("banner");
  if (ok){
    connState = {ok:true, lastOk:Date.now()};
    pill.className = "pill ok";
    pill.innerHTML = '<span class="dot"></span>已连接';
    banner.classList.add("hidden");
  } else {
    if (!connState.lastOk) connState.lastOk = Date.now();
    connState.ok = false;
    pill.className = "pill off";
    pill.innerHTML = '<span class="dot"></span>连接中断';
    var last = connState.lastOk
      ? new Date(connState.lastOk).toLocaleTimeString("zh-CN", {hour12:false}) : "从未";
    banner.classList.remove("hidden");
    banner.textContent = "与面板服务连接中断（" + String(err || "") + "），保留最后数据 " + last;
  }
}

function fmtUptime(sec){
  if (sec == null || isNaN(sec)) return "—";
  sec = Math.floor(Number(sec));
  var h = Math.floor(sec/3600), m = Math.floor(sec%3600/60), s = sec%60;
  return (h ? h+"h " : "") + m + "m " + s + "s";
}

function fmtDur(sec){
  if (sec == null || isNaN(sec)) return "—";
  sec = Math.max(0, Math.floor(Number(sec)));
  if (sec < 60) return sec + "s";
  var m = Math.floor(sec/60), s = sec % 60;
  if (m < 60) return m + "m" + (s ? " " + s + "s" : "");
  return Math.floor(m/60) + "h" + (m % 60 ? " " + (m % 60) + "m" : "");
}

function isLlamaProc(n){ return /llama|vllm|ollama|gpt/i.test(String(n || "")); }

function gpu0(h){
  return (h && h.gpu && h.gpu.gpus && h.gpu.gpus[0]) || null;
}

function histVals(fn){
  var out = [];
  for (var i = 0; i < histCache.length; i++){
    var v = fn(histCache[i]);
    if (v != null && !isNaN(v)) out.push(Number(v));
  }
  return out;
}

function fitCanvas(cv){
  var dpr = window.devicePixelRatio || 1;
  var W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return null;
  cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr);
  var c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  return {c:c, W:W, H:H};
}

function drawSpark(id, vals, color){
  var cv = $(id); if (!cv) return;
  var f = fitCanvas(cv); if (!f) return;
  var c = f.c, W = f.W, H = f.H;
  if (vals.length < 2) return;
  var mn = Infinity, mx = -Infinity;
  for (var i = 0; i < vals.length; i++){
    if (vals[i] < mn) mn = vals[i];
    if (vals[i] > mx) mx = vals[i];
  }
  if (mx - mn < 1e-6) mx = mn + 1;
  var pad = (mx - mn) * 0.12; mn -= pad; mx += pad;
  c.strokeStyle = color; c.lineWidth = 1.5; c.lineJoin = "round";
  c.beginPath();
  for (i = 0; i < vals.length; i++){
    var x = i / (vals.length - 1) * W;
    var y = H - 2 - (vals[i] - mn) / (mx - mn) * (H - 4);
    i ? c.lineTo(x, y) : c.moveTo(x, y);
  }
  c.stroke();
  c.lineTo(W, H); c.lineTo(0, H); c.closePath();
  c.globalAlpha = 0.08; c.fillStyle = color; c.fill(); c.globalAlpha = 1;
}

function drawChart(hist){
  var cv = $("chart"); if (!cv) return;
  var f = fitCanvas(cv); if (!f) return;
  var c = f.c, W = f.W, H = f.H;
  var L = 30, R = 8, T = 10, B = 18;
  var pw = W - L - R, ph = H - T - B;
  c.strokeStyle = "rgba(148,163,184,.14)";
  c.fillStyle = "#5b6884";
  c.font = "10px sans-serif";
  c.lineWidth = 1;
  var y;
  [0,25,50,75,100].forEach(function(vv){
    y = T + ph - vv/100*ph;
    c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw, y); c.stroke();
    c.fillText(String(vv), 4, y + 3);
  });
  var n = hist.length;
  if (n > 1){
    for (var k = 0; k <= 4; k++){
      var idx = Math.round(k/4*(n-1));
      var t = String(hist[idx].ts || "").slice(11, 16);
      var x = L + idx/(n-1)*pw;
      var tx = (k === 0) ? x : (k === 4 ? x - 34 : x - 17);
      c.fillText(t, tx, H - 4);
    }
    var series = [
      {fn:function(h){ var g = gpu0(h); return g ? g.util : null; }, color:COLORS.gpu, fill:true},
      {fn:function(h){ return (h.ram_pct != null) ? h.ram_pct : null; }, color:COLORS.ram},
      {fn:function(h){ return (h.cpu != null) ? h.cpu : null; }, color:COLORS.cpu}
    ];
    for (var si = 0; si < series.length; si++){
      var s = series[si], pts = [];
      for (var j = 0; j < n; j++){
        var val = s.fn(hist[j]);
        if (val != null && !isNaN(val)) pts.push([j, Number(val)]);
      }
      if (pts.length < 2) continue;
      c.strokeStyle = s.color; c.lineWidth = 1.8; c.lineJoin = "round"; c.lineCap = "round";
      c.beginPath();
      for (var pi = 0; pi < pts.length; pi++){
        var px = L + pts[pi][0]/(n-1)*pw;
        var py = T + ph - Math.max(0, Math.min(100, pts[pi][1]))/100*ph;
        pi ? c.lineTo(px, py) : c.moveTo(px, py);
      }
      c.stroke();
      if (s.fill){
        c.lineTo(L + pts[pts.length-1][0]/(n-1)*pw, T + ph);
        c.lineTo(L + pts[0][0]/(n-1)*pw, T + ph);
        c.closePath();
        c.globalAlpha = 0.07; c.fillStyle = s.color; c.fill(); c.globalAlpha = 1;
      }
    }
  } else {
    c.fillStyle = "#5b6884";
    c.fillText("采集中，历史不足…", L + 8, T + ph/2);
  }
  var last = hist[n-1] || {};
  var g0 = gpu0(last) || {};
  $("legend").innerHTML =
    chip(COLORS.gpu, "GPU", (g0.util != null) ? Math.round(g0.util) + " %" : "—") +
    chip(COLORS.ram, "内存", (last.ram_pct != null) ? Math.round(last.ram_pct) + " %" : "—") +
    chip(COLORS.cpu, "CPU", (last.cpu != null) ? Math.round(last.cpu) + " %" : "—");
}
function chip(color, name, val){
  return '<span class="chip"><i style="background:' + color + '"></i>' + name + " <b>" + val + "</b></span>";
}

function slotRow(sl){
  var st = sl.processing
    ? '<span class="st busy"><span class="dot"></span>生成中</span>'
    : '<span class="st idle"><span class="dot"></span>空闲</span>';
  var cp = clampPct(sl.ctx_pct); if (cp == null) cp = 0;
  var ccol = cp > 80 ? "var(--err)" : (cp > 50 ? "var(--warn)" : "var(--ram)");
  var gr = (sl.processing && sl.gen_recent_tps != null)
    ? '<b>' + sl.gen_recent_tps + "</b> t/s" : "—";
  var ga = (sl.processing && sl.gen_avg_tps != null)
    ? sl.gen_avg_tps + " t/s" : "—";
  return "<tr>" +
    "<td>" + esc(sl.id) + "</td>" +
    "<td>" + st + "</td>" +
    '<td class="mono">' + esc(sl.task == null ? "" : sl.task) + "</td>" +
    "<td>" + gr + "</td>" +
    "<td>" + ga + "</td>" +
    "<td>" + (sl.prompt_tps != null ? sl.prompt_tps + " t/s" : "—") + "</td>" +
    "<td>" + fmtDur(sl.task_elapsed) + "</td>" +
    "<td>" + (sl.remain_est != null ? "≈ " + fmtDur(sl.remain_est) : "—") + "</td>" +
    "<td>" + fmt(sl.n_decoded, 0) + "</td>" +
    '<td><div class="ctx"><span class="ctxbar"><i style="width:' + cp + "%;background:" + ccol +
      '"></i></span><span>' + fmt(sl.ctx_used, 0) + " / " + fmt(sl.ctx, 0) +
      (sl.ctx_pct != null ? " (" + sl.ctx_pct + "%)" : "") + "</span></div></td>" +
    "<td>" + (sl.speculative ? "MTP" : "—") + "</td>" +
    "</tr>";
}

function render(s){
  lastData = s;
  var stale = (Date.now() - new Date(s.ts).getTime()) > 20000;
  $("ts").textContent = s.ts + (stale ? "（数据已陈旧）" : "");
  $("uptime").textContent = "运行 " + fmtUptime(s.uptime_s);

  var gpu = s.gpu || {};
  var gpus = (gpu.gpus && gpu.gpus.length) ? gpu.gpus : [];
  var g0 = gpus[0] || {};

  // 优雅降级：没采到任何 GPU（未装 NVIDIA 卡 / nvidia-smi 失败）时隐藏 GPU 卡片，
  // 剩下内存 / CPU / 磁盘照常显示
  var noGpu = !gpus.length;
  ["card_gpu_util", "card_gpu_mem", "card_gpu_temp"].forEach(function(id){
    $(id).classList.toggle("hidden", noGpu);
  });
  $("gname").textContent = g0.name || (noGpu ? "未检测到 NVIDIA GPU" : "NVIDIA GPU");

  // GPU 利用率
  var util = g0.util;
  $("gpu_util").innerHTML = (util == null) ? 'N/A <small class="err">采集失败</small>'
                                           : Math.round(util) + " <small>%</small>";
  setBar("gpu_util_bar", util);
  drawSpark("sp_util", histVals(function(h){ var g = gpu0(h); return g ? g.util : null; }), COLORS.gpu);

  // 显存
  var vu = g0.mem_used, vt = g0.mem_total;
  var vp = (vu != null && vt) ? vu/vt*100 : null;
  $("gpu_mem").innerHTML = (vu == null) ? "N/A"
    : fmt(vu/1024, 1) + " <small>/ " + fmt(vt/1024, 0) + " GB</small>";
  $("gpu_mem_extra").textContent = (vp == null) ? "" : "显存利用率 " + Math.round(vp) + " %";
  setBar("gpu_mem_bar", vp);
  drawSpark("sp_vram", histVals(function(h){
    var g = gpu0(h);
    return (g && g.mem_used != null && g.mem_total) ? g.mem_used/g.mem_total*100 : null;
  }), COLORS.gpu);

  // 温度 / 功耗
  $("gpu_temp").innerHTML = (g0.temp == null) ? "N/A" : Math.round(g0.temp) + " <small>°C</small>";
  var fanTxt = (g0.fan != null) ? " · 风扇 " + Math.round(g0.fan) + "%" : "";
  $("gpu_extra").textContent = ((g0.power != null) ? "功耗 " + Math.round(g0.power) + " W" : "") + fanTxt;
  drawSpark("sp_temp", histVals(function(h){ var g = gpu0(h); return g ? g.temp : null; }), COLORS.temp);

  // 内存
  $("ram").innerHTML = fmt(s.ram_used_mb/1024, 1) + " <small>/ " + fmt(s.ram_total_mb/1024, 0) + " GB</small>";
  $("ram_extra").textContent = Math.round(s.ram_pct) + " %";
  setBar("ram_bar", s.ram_pct);
  drawSpark("sp_ram", histVals(function(h){ return (h.ram_pct != null) ? h.ram_pct : null; }), COLORS.ram);

  // CPU
  $("cpu").innerHTML = Math.round(s.cpu) + " <small>%</small>";
  setBar("cpu_bar", s.cpu);
  drawSpark("sp_cpu", histVals(function(h){ return (h.cpu != null) ? h.cpu : null; }), COLORS.cpu);

  // 磁盘
  var d = s.disk || {};
  if (d.total_gb){
    $("disk_label").textContent = "磁盘 " + (d.label || "");
    $("disk").innerHTML = fmt(d.free_gb, 1) + " <small>/ " + fmt(d.total_gb, 0) + " GB</small>";
    $("disk_extra").textContent = "已用 " + d.pct + " %";
    setBar("disk_bar", d.pct);
  }

  // 多 GPU
  var mg = $("multi_gpu_panel");
  if (gpus.length > 1){
    mg.classList.remove("hidden");
    $("gpus_tbody").innerHTML = gpus.map(function(g){
      return "<tr><td>" + esc(g.name) + "</td>" +
        "<td>" + (g.util == null ? "—" : Math.round(g.util) + "%") + "</td>" +
        "<td>" + ((g.mem_used != null) ? fmt(g.mem_used/1024, 1) + " / " + fmt(g.mem_total/1024, 0) + " GB" : "—") + "</td>" +
        "<td>" + (g.temp == null ? "—" : Math.round(g.temp) + "°C") + "</td>" +
        "<td>" + (g.power == null ? "—" : Math.round(g.power) + " W") + "</td>" +
        "<td>" + (g.fan == null ? "—" : Math.round(g.fan) + "%") + "</td></tr>";
    }).join("");
  } else {
    mg.classList.add("hidden");
  }

  // llama
  var L = s.llama || {};
  var lp = $("llama_pill");
  if (L.error){
    lp.className = "pill off";
    lp.innerHTML = '<span class="dot"></span>离线';
  } else {
    lp.className = "pill ok";
    lp.innerHTML = '<span class="dot"></span>在线';
  }
  $("llama_meta").textContent = (L.error ? "" : "忙碌 slot " + L.busy + " / " + L.total + " · ") + (L.url || "");
  var busy = (L.error) ? 0 : (L.busy || 0);
  var chips = chip("#5eead4", "总输出速度", (busy && L.tot_recent_tps != null) ? L.tot_recent_tps + " t/s" : "—")
    + chip("#5eead4", "总平均速度", (busy && L.tot_avg_tps != null) ? L.tot_avg_tps + " t/s" : "—")
    + chip("#ffd166", "总 Prompt", (L.tot_prompt_tps != null && L.tot_prompt_tps > 0) ? L.tot_prompt_tps + " t/s" : "—")
    + chip("#7db8ff", "观察生成", fmt(L.sess_decoded, 0) + " tok");
  $("llama_chips").innerHTML = L.error ? "" : chips;
  drawSpark("sp_llama", histVals(function(h){
    var ll = h.llama;
    return (ll && ll.tot_recent_tps != null) ? ll.tot_recent_tps : null;
  }), "#5eead4");
  var tb = $("slots_tbody");
  if (L.error){
    tb.innerHTML = '<tr><td colspan="11" class="err">连接失败: ' + esc(L.error) + "</td></tr>";
  } else if (!L.slots || !L.slots.length){
    tb.innerHTML = '<tr><td colspan="11" class="dim">无 slot</td></tr>';
  } else {
    tb.innerHTML = L.slots.map(slotRow).join("");
  }

  // 进程
  var procs = s.top_procs || [];
  var maxMem = procs.length ? procs[0].mem_mb : 0;
  $("procs_tbody").innerHTML = procs.map(function(p){
    return '<tr><td data-th="PID">' + p.pid + "</td>" +
      '<td data-th="进程名">' + esc(p.name) + (isLlamaProc(p.name) ? '<span class="tag">推理</span>' : "") + "</td>" +
      '<td data-th="内存 MB"><div class="memcell"><span class="membar"><i style="width:' +
      (maxMem ? p.mem_mb/maxMem*100 : 0) + '%"></i></span><b>' + fmt(p.mem_mb, 0) + "</b></div></td></tr>";
  }).join("");

  // 主趋势图（V2 漏调用的 bug，V3 修复）
  drawChart(histCache);

  // 采样记录
  $("log_tbody").innerHTML = histCache.slice(-15).reverse().map(function(h){
    var g = gpu0(h) || {};
    return "<tr><td>" + esc(h.ts) + "</td><td>" + fmt(g.util, 0) + "</td><td>" + fmt(g.mem_used, 0) +
      "</td><td>" + fmt(g.temp, 0) + "</td><td>" + fmt(h.ram_pct, 0) + "</td><td>" + fmt(h.cpu, 0) + "</td></tr>";
  }).join("");
}

/* ---------------- 工作统计 ---------------- */
var STATS_W = ["24h", "7d", "30d", "1y"];
var STATS_LABEL = {"24h":"24 小时","7d":"7 天","30d":"30 天","1y":"1 年"};
var STATS_RANGE_S = {"24h":86400,"7d":604800,"30d":2592000,"1y":31536000};
var statsData = null;
var statsRange = "24h";
var statsMetric = "tok";
var statsGeom = null;

function fmtTok(v){
  if (v == null || isNaN(v)) return "—";
  v = Number(v);
  if (v >= 1e9) return (v/1e9).toFixed(2) + " G";
  if (v >= 1e6) return (v/1e6).toFixed(2) + " M";
  if (v >= 1e4) return (v/1e3).toFixed(1) + " k";
  return fmt(v, 0);
}

function fetchStats(){
  fetch("/api/stats" + Q, {cache:"no-store"})
    .then(function(r){ return r.ok ? r.json() : null; })
    .then(function(d){ if (d){ statsData = d; renderStats(); } })
    .catch(function(){});
}

function statCard(color, label, val, sub){
  return '<div class="card" style="--c:' + color + '"><div class="label">' + esc(label) +
    '</div><div class="value">' + val + '</div>' +
    (sub ? '<div class="extra">' + sub + '</div>' : '') + '</div>';
}

function renderStats(){
  var d = statsData;
  if (!d || !d.ready){
    $("stats_cards").innerHTML = '<div class="dim" style="padding:8px">统计初始化中…</div>';
    return;
  }
  $("stats_meta").textContent = "数据自 " + (d.data_from || "—") + " 开始记录 · 更新于 " + (d.updated_at || "—");
  var w = (d.windows || {})[statsRange] || {};
  var rangeS = STATS_RANGE_S[statsRange];
  var pctOf = function(s){
    return (s != null && rangeS) ? "占 " + Math.round(s/rangeS*1000)/10 + " %" : "";
  };
  var tabs = $("stats_tabs");
  tabs.innerHTML = "";
  STATS_W.forEach(function(k){
    var b = document.createElement("button");
    b.className = "stab" + (k === statsRange ? " on" : "");
    b.textContent = STATS_LABEL[k];
    b.onclick = function(){ statsRange = k; renderStats(); };
    tabs.appendChild(b);
  });
  var mt = $("stats_metric");
  mt.innerHTML = "";
  [["tok", "Token"], ["busy", "推理时长"]].forEach(function(p){
    var b = document.createElement("button");
    b.className = "stab" + (statsMetric === p[0] ? " on" : "");
    b.textContent = p[1];
    b.onclick = function(){ statsMetric = p[0]; renderStats(); };
    mt.appendChild(b);
  });
  var w = (d.windows || {})[statsRange] || {};
  var rangeS = STATS_RANGE_S[statsRange];
  var pctOf = function(s){
    return (s != null && rangeS) ? "占 " + Math.round(s/rangeS*1000)/10 + " %" : "";
  };
  var cacheSub = "";
  if (w.tok_cached != null && (w.tok_prompt != null) && (w.tok_prompt + w.tok_cached) > 0){
    var hitPct = Math.round(w.tok_cached / (w.tok_prompt + w.tok_cached) * 1000) / 10;
    cacheSub = "缓存命中 " + fmtTok(w.tok_cached) + " · " + hitPct + " %";
  }
  $("stats_cards").innerHTML =
    statCard("#ffd166", "输入 token",
      (w.tok_prompt != null ? fmtTok((w.tok_prompt || 0) + (w.tok_cached || 0)) : "—"),
      cacheSub || "实际计算 " + fmtTok(w.tok_prompt)) +
    statCard("#5eead4", "生成 token", fmtTok(w.tok_gen), (w.avg_tps ? "平均 " + w.avg_tps + " t/s" : "")) +
    statCard("#5eead4", "投机接受率", (w.spec_acc != null ? w.spec_acc + " %" : "—"), "MTP 草稿命中") +
    statCard("#7db8ff", "推理忙碌", fmtDur(w.busy_s), pctOf(w.busy_s)) +
    statCard("#ff7a59", "GPU 活跃", fmtDur(w.gpu_active_s), pctOf(w.gpu_active_s)) +
    statCard("#7ee081", "平均 GPU 利用", (w.gpu_util_avg != null ? w.gpu_util_avg + " %" : "—"), "") +
    statCard("#c792ea", "峰值温度", (w.temp_max != null ? Math.round(w.temp_max) + " °C" : "—"), "");
  var buckets;
  if (statsRange === "24h") buckets = (d.buckets || {}).hourly;
  else if (statsRange === "1y") buckets = (d.buckets || {}).weekly;
  else buckets = ((d.buckets || {}).daily || []).slice(statsRange === "7d" ? -7 : -30);
  drawStatsChart(buckets || [], statsMetric);
}

function drawStatsChart(buckets, metric){
  var f = fitCanvas($("stats_chart")); if (!f) return;
  var c = f.c, W = f.W, H = f.H, L = 44, R = 8, T = 12, B = 20;
  var pw = W - L - R, ph = H - T - B;
  c.strokeStyle = "rgba(148,163,184,.14)";
  c.fillStyle = "#5b6884";
  c.font = "10px sans-serif";
  c.lineWidth = 1;
  var isTok = (metric === "tok");
  var fmtV = isTok ? fmtTok : function(v){ return fmtDur(v); };
  var mx = 0;
  buckets.forEach(function(b){
    var v = isTok ? (b.tok || 0) : (b.busy_s || 0);
    if (v > mx) mx = v;
  });
  [0, 0.5, 1].forEach(function(v){
    var y = T + ph - v*ph;
    c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw, y); c.stroke();
    c.fillText(fmtV(mx*v), 4, y + 3);
  });
  var n = buckets.length;
  statsGeom = n ? {L:L, bw:pw/n, n:n, buckets:buckets} : null;
  if (!n) return;
  if (mx <= 0){
    c.fillText("该时段暂无推理数据", L + 8, T + ph/2);
  }
  var bw = pw / n;
  buckets.forEach(function(b, i){
    var v = isTok ? (b.tok || 0) : (b.busy_s || 0);
    var h = (v > 0 && mx > 0) ? Math.max(2, v/mx*ph) : 0;
    c.fillStyle = (v > 0) ? "rgba(255,122,89,.85)" : "rgba(148,163,184,.15)";
    c.fillRect(L + i*bw + 1, T + ph - h, Math.max(1, bw - 2), h);
  });
  var step = Math.max(1, Math.ceil(n/6));
  for (var i = 0; i < n; i += step){
    c.fillStyle = "#5b6884";
    c.fillText(buckets[i].label, L + i*bw, H - 4);
  }
}

/* 柱状图悬停提示 */
(function(){
  var cv = $("stats_chart"), tip = $("stats_tip");
  if (!cv || !tip) return;
  cv.addEventListener("mousemove", function(e){
    if (!statsGeom){
      tip.style.display = "none";
      return;
    }
    var r = cv.getBoundingClientRect();
    var i = Math.floor((e.clientX - r.left - statsGeom.L) / statsGeom.bw);
    if (i < 0 || i >= statsGeom.n){
      tip.style.display = "none";
      return;
    }
    var b = statsGeom.buckets[i];
    tip.innerHTML = "<b>" + esc(b.label) + "</b><br>" +
      "生成 " + fmtTok(b.tok) + " tok<br>" +
      "推理忙碌 " + fmtDur(b.busy_s);
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 180) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  cv.addEventListener("mouseleave", function(){ tip.style.display = "none"; });
})();

function initHealth(){
  fetch("/api/health" + Q, {cache:"no-store"})
    .then(function(r){ if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(function(h){
      if (h && h.interval) $("intv").textContent = String(Math.round(Number(h.interval)));
    })
    .catch(function(){});
}

async function tick(){
  try {
    var ctl = new AbortController();
    var to = setTimeout(function(){ ctl.abort(); }, 5000);
    var rs = await Promise.all([
      fetch("/api/latest" + Q, {signal:ctl.signal, cache:"no-store"}),
      fetch("/api/history?limit=900" + (TOKEN ? "&token=" + encodeURIComponent(TOKEN) : ""), {signal:ctl.signal, cache:"no-store"})
    ]);
    clearTimeout(to);
    if (!rs[0].ok) throw new Error("HTTP " + rs[0].status);
    var s = await rs[0].json();
    if (!s || !s.ts) throw new Error("数据为空");
    if (rs[1].ok) histCache = await rs[1].json();
    render(s);
    setConn(true);
  } catch(e){
    setConn(false, (e && e.name === "AbortError") ? "超时" : (e && e.message || String(e)));
  }
}

initHealth();
setInterval(tick, 2000);
tick();
fetchStats();
setInterval(fetchStats, 60000);
window.addEventListener("resize", function(){
  if (lastData) render(lastData);
  if (statsData) renderStats();
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "gpuPanel/4.0"

    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self):
        if not PANEL_TOKEN:
            return True
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if q.get("token", [""])[0] == PANEL_TOKEN:
            return True
        return self.headers.get("Authorization", "") == "Bearer " + PANEL_TOKEN

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return
        if not self._authorized():
            self._send(403, "forbidden", "text/plain")
            return
        if p == "/api/latest":
            with _history_lock:
                latest = list(_history)[-1:]
            body = latest[0] if latest else {}
            self._send(200, json.dumps(body, ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif p == "/api/history":
            limit = 60
            try:
                limit = int(parse_qs(u.query).get("limit", ["60"])[0])
                limit = max(1, min(limit, MAX_HISTORY))
            except ValueError:
                pass
            with _history_lock:
                items = list(_history)[-limit:]
            # 历史列表不带 top_procs，减小体积
            slim = [{k: v for k, v in s.items() if k != "top_procs"} for s in items]
            self._send(200, json.dumps(slim, ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif p == "/api/stats":
            with _stats_lock:
                body = dict(_stats_cache)
            self._send(200, json.dumps(body, ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif p == "/api/health":
            self._send(200, json.dumps({
                "ok": True,
                "version": 4,
                "uptime_s": int(time.time() - START_TIME),
                "samples": len(_history),
                "interval": SAMPLE_INTERVAL,
                "llm_url": LLM_URL,
            }, ensure_ascii=False), "application/json; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")


def main():
    import sys
    global PORT
    if len(sys.argv) > 1:
        try:
            PORT = int(sys.argv[1])
        except ValueError:
            pass
    threading.Thread(target=sampler_loop_fast, daemon=True).start()
    threading.Thread(target=sampler_loop_slow, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    print("监控面板 V4 已启动: http://127.0.0.1:%d  (局域网: http://<本机IP>:%d)" % (PORT, PORT))
    if CSV_INTERVAL <= 0:
        print("LLM: %s   采样 %.1fs   历史 %d 条   CSV: 已关闭" % (LLM_URL, SAMPLE_INTERVAL, MAX_HISTORY))
    else:
        print("LLM: %s   采样 %.1fs   历史 %d 条   CSV: %s (每 %gs 一行)" % (LLM_URL, SAMPLE_INTERVAL, MAX_HISTORY, CSV_LOG, CSV_INTERVAL))
    if PANEL_TOKEN:
        print("认证已启用 (PANEL_TOKEN)")
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    except OSError as e:
        # 端口被占用时（比如旧实例还在后台跑），无窗口启动会悄悄失败，
        # 这里弹一次提示让用户知道原因。
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0, "端口 %d 已被占用，可能已有面板实例在运行。\n\n%s" % (PORT, e),
                    "gpu_panel_v4", 0x10)
            except Exception:
                pass
        raise
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
