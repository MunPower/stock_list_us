import os
# 행렬 계산이 CPU 코어를 전부 차지하지 않도록 계산 스레드를 1개로 제한 (numpy를 불러오기 전에 설정해야 함)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import requests
import json
import time
import threading
import html
from concurrent.futures import ThreadPoolExecutor
import re
import streamlit.components.v1 as components
from pathlib import Path
from urllib.parse import urlencode

st.set_page_config(page_title="미국주식 거래대금 TOP30 차트", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding:0.8rem 0.4rem 110px 0.4rem;max-width:100%;}  /* 아래 110px 여백: GitHub 화면(index.html)이 아래 72px를 가려도 마지막 줄까지 보이게 */
header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,
[data-testid="manage-app-button"],[class*="_profileContainer_"],[class*="viewerBadge"],[class*="_terminalButton_"],[class*="_gzau3_"]{display:none !important;}
</style>""", unsafe_allow_html=True)

# ───────────── 차트 계산 설정(고정값) ─────────────
disp, calc, lam, kmul, band_pct = 120, 720, 8e3, 1.6, 0.0


TOP_N = 30   # 거래대금 상위 몇 등까지 보여줄지
UA = {"User-Agent": "Mozilla/5.0"}


def _num(v):
    """'1,234' / '-1.5' / NaN 등을 숫자(또는 None)로"""
    try:
        f = float(str(v).replace(",", "").replace("%", "").strip())
        return f if np.isfinite(f) else None
    except Exception:
        return None


def yticker(t: str) -> str:
    """BRK.B → BRK-B (야후·FinanceDataReader 표기)"""
    return str(t).strip().upper().replace(".", "-").replace("/", "-")


def _row(code, name, close, chg, ratio, opn, high, low, tval, ex):
    """리스트 한 줄: [티커, 이름, 현재가, 대비(부호 포함), 등락률, 시가, 고가, 저가, 거래대금($), 거래소]"""
    c, r, d = _num(close), _num(ratio), _num(chg)
    if d is not None and r is not None:      # 대비의 부호는 등락률 부호로 맞춤(상승 +, 하락 -)
        d = abs(d) * (1 if r > 0 else (-1 if r < 0 else 0))
    return [yticker(code), str(name), c, d, r, _num(opn), _num(high), _num(low), _num(tval), str(ex or "")]


# ───────────── 거래대금 순위 받기 ─────────────
# ① TradingView 종목 검색(거래대금 Value.Traded 순) → ② 안 되면 대형주 후보를 직접 받아 (종가×거래량)으로 순위
def _from_tradingview(n):
    body = {
        "markets": ["america"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "options": {"lang": "en"},
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},      # 일반 주식·ADR (ETF 제외)
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
            {"left": "is_primary", "operation": "equal", "right": True},
        ],
        "columns": ["name", "description", "close", "change_abs", "change", "open", "high", "low",
                    "Value.Traded", "exchange"],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, n],
    }
    r = requests.post("https://scanner.tradingview.com/america/scan", json=body, headers=UA, timeout=8)
    rows = []
    for it in r.json().get("data", [])[:n]:
        d = it.get("d", [])
        if len(d) < 10:
            continue
        rows.append(_row(d[0], d[1] or d[0], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]))
    if len(rows) < n // 2:
        raise ValueError("빈 목록")
    return rows


# 거래대금이 큰 편인 미국 대형주 후보 (①이 안 될 때만 사용)
POOL = ("NVDA TSLA AAPL MSFT AMZN META GOOGL GOOG AVGO AMD PLTR NFLX MSTR COIN SMCI MU INTC UBER ORCL CRM "
        "LLY UNH JPM V MA BAC XOM WMT COST HD ADBE QCOM ARM TSM BABA PDD SHOP SNOW PANW CRWD ABNB PYPL "
        "DIS BA NKE KO PEP CVX GS MS WFC C CAT GE IBM TXN AMAT LRCX KLAC MRVL ASML NVO RIVN SOFI HOOD "
        "APP RKLB IONQ ANET DELL VRT NOW INTU ISRG BRK-B JNJ PG ABBV MRK").split()
POOL_EX = {}   # 후보 종목의 거래소(모르면 비워 둠 → 지수는 S&P500)


def _yahoo_daily(sym: str, start: str):
    """야후 차트 주소에서 일봉 받기 (FinanceDataReader가 안 될 때 사용)"""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(time.time()) + 86400
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"period1": p1, "period2": p2, "interval": "1d", "includePrePost": "false"},
                     headers=UA, timeout=10)
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    tz = res.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    d = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                      "Close": q["close"], "Volume": q["volume"]}, index=idx, dtype=float)
    d = d[~d.index.duplicated(keep="last")].dropna(subset=["Close"])
    return d


def _daily(sym: str, start: str):
    """일봉: FinanceDataReader 먼저, 안 되면 야후에서 직접"""
    try:
        d = fdr.DataReader(sym, start)
        if d is not None and len(d) >= 5 and "Close" in d.columns:
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize() if getattr(d.index, "tz", None) \
                else pd.to_datetime(d.index).normalize()
            return d
    except Exception:
        pass
    return _yahoo_daily(sym, start)


def _from_pool(n):
    start = (pd.Timestamp.today() - pd.Timedelta(days=12)).strftime("%Y-%m-%d")

    def one(t):
        try:
            d = _daily(t, start).dropna(subset=["Close"])
            if len(d) < 2:
                return None
            a, b = d.iloc[-1], d.iloc[-2]
            chg = a["Close"] - b["Close"]
            return _row(t, t, a["Close"], chg, chg / b["Close"] * 100, a.get("Open"), a.get("High"),
                        a.get("Low"), a["Close"] * a.get("Volume", 0), POOL_EX.get(t, ""))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(one, POOL) if r and r[8]]
    rows.sort(key=lambda r: r[8], reverse=True)
    if not rows:
        raise ValueError("빈 목록")
    return rows[:n]


@st.cache_data(ttl=600, show_spinner=False)
def top_list(n: int = TOP_N):
    """미국 주식 거래대금 1~n위. 10분 동안은 저장해 둔 목록을 그대로 사용 (새로고침 버튼으로 바로 다시 받기)"""
    asof = pd.Timestamp.now(tz="Asia/Seoul").strftime("%m-%d %H:%M")
    for f in (_from_tradingview, _from_pool):
        try:
            return f(n), asof
        except Exception:
            continue
    return [], asof


# ───────────── 공통: 시세 받기 + 매도선·매수선 계산 (리스트 신호와 차트가 똑같이 사용) ─────────────


def _start(days: int) -> str:
    return (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")


STORE_TTL = 600     # 받은 시세를 10분 동안 재사용


@st.cache_resource(show_spinner=False)
def _store():
    """리스트에서 받은 종목 시세를 차트도 그대로 쓰도록 서버에 보관하는 저장소"""
    return {"lock": threading.Lock(), "data": {}}


@st.cache_resource(show_spinner=False)
def _sig_store():
    """종목별 신호 보관함 {(코드, 기간): (그 시세를 받은 시각, 신호)}"""
    return {}


def _hist_time(code: str, days: int):
    ent = _store()["data"].get((code, int(days)))
    return ent[0] if ent and time.time() - ent[0] < STORE_TTL else None


def known_signals(codes: tuple, days: int) -> dict:
    """시세도 신호도 이미 보관돼 있는 종목만 {코드: 신호} (받기·계산 없이 바로)"""
    out, ss_ = {}, _sig_store()
    for c in codes:
        t = _hist_time(c, days)
        ent = ss_.get((c, int(days)))
        if t is not None and ent and ent[0] == t:
            out[c] = ent[1]
    return out


def cached_hist(code: str, days: int):
    """저장소에 10분 이내에 받은 시세가 있으면 그것을, 없으면 None"""
    ent = _store()["data"].get((code, int(days)))
    if ent and time.time() - ent[0] < STORE_TTL:
        return ent[1]
    return None


def _fetch(code: str, days: int):
    d = _daily(code, _start(days)).tail(int(days))
    st_ = _store()
    with st_["lock"]:
        st_["data"][(code, int(days))] = (time.time(), d)
    return d


def hist(code: str, days: int):
    """종목 일봉 (10분 동안 저장해 두고 리스트·차트가 같이 사용)"""
    d = cached_hist(code, days)
    return d if d is not None else _fetch(code, days)


IDX_SYMS = {"나스닥": ("IXIC", "^IXIC"), "S&P500": ("US500", "^GSPC")}


@st.cache_data(ttl=600, show_spinner=False)
def idx_hist(idx_name: str, days: int):
    """지수 일봉: FinanceDataReader 기호 → 안 되면 야후 기호"""
    a, b = IDX_SYMS[idx_name]
    for s in (a, b):
        try:
            d = _daily(s, _start(days))
            if d is not None and len(d) >= 60:
                return d.tail(int(days))
        except Exception:
            continue
    return _yahoo_daily(b, _start(days)).tail(int(days))


@st.cache_resource(show_spinner=False, max_entries=32)
def _hp_inv(n: int, lam: float) -> np.ndarray:
    """HP필터 계산용 행렬의 역행렬. 기간(n)이 같으면 모든 종목이 같은 행렬이라 한 번만 만들어
    서버에 계속 보관 (화면을 새로 열어도 다시 만들지 않음)"""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.inv(np.eye(n) + lam * D.T @ D)


def hp_trend(z: np.ndarray, lam: float) -> np.ndarray:
    """Hodrick-Prescott 필터의 추세 성분: (I + λ D'D) τ = z"""
    return _hp_inv(len(z), float(lam)) @ z


def calc_bands(y: np.ndarray):
    """곡선추세, 매도선(상한), 매수선(하한)"""
    t = np.arange(len(y))
    b1, b0 = np.polyfit(t, y, 1)              # 직선추세: 계산기간 전체 1차 회귀
    line = b0 + b1 * t
    curve = line + hp_trend(y - line, lam)    # 곡선추세: 직선추세 제거 → HP필터 → 복원
    width = band_pct / 100 if band_pct > 0 else kmul * (y / curve - 1).std()
    return curve, curve * (1 + width), curve * (1 - width)


TOUCH_TOL = 0          # 0 = 매도선·매수선에 정확히 닿거나 넘었을 때만 신호


def signal_of(y: np.ndarray):
    """'sell'(매도 가능) / 'buy'(매수 가능) / None"""
    _, up, lo = calc_bands(y)
    last = y[-1]
    if last >= up[-1] * (1 - TOUCH_TOL):
        return "sell"
    if last <= lo[-1] * (1 + TOUCH_TOL):
        return "buy"
    return None


def prefetch(codes: tuple, days: int, timeout: float = 40):
    """저장소에 없는(또는 10분 지난) 종목만 동시에 받아 저장. 전체 40초가 지나면 받은 것까지만 사용"""
    from concurrent.futures import wait
    missing = [c for c in codes if cached_hist(c, days) is None]
    if missing:
        ex = ThreadPoolExecutor(max_workers=8)
        futs = [ex.submit(_fetch, c, days) for c in missing]
        wait(futs, timeout=timeout)
        ex.shutdown(wait=False, cancel_futures=True)
    return {c: cached_hist(c, days) for c in codes}


def fmt_sec(x: float) -> str:
    return f"{x:.1f}초" if x < 10 else f"{x:.0f}초"


def timing_html(parts) -> str:
    """[('리스트', 0.3), ('시세', 5.2), ...] → 작은 회색 글씨 한 줄. 느린 칸(3초 이상)은 주황색"""
    out = []
    for k, v in parts:
        col = "#d9480f" if v >= 3 else "#888"
        out.append(f'{k} <b style="color:{col}">{fmt_sec(v)}</b>')
    return '<span class="tm">⏱ ' + " · ".join(out) + "</span>"


def list_signals(codes: tuple):
    """리스트 종목들의 매수·매도 신호 {코드: 'buy'/'sell'/None}, 시세를 받은 종목 수"""
    days = max(int(calc), 180)
    _t = time.time()
    data = prefetch(codes, days)
    t_fetch = time.time() - _t
    _t = time.time()
    sig, got = {}, 0
    known = known_signals(codes, days)
    ss_ = _sig_store()
    for c in codes:
        d = data.get(c)
        sig[c] = None
        if d is None:
            continue
        got += 1
        if c in known:                    # 같은 시세로 이미 계산해 둔 신호
            sig[c] = known[c]
            continue
        try:
            y = d["Close"].to_numpy(dtype=float)
            if len(y) >= 60 and np.isfinite(y).all():
                sig[c] = signal_of(y)
        except Exception:
            pass
        t = _hist_time(c, days)
        if t is not None:
            ss_[(c, int(days))] = (t, sig[c])
    return sig, got, t_fetch, time.time() - _t


# ───────────── ① 리스트 화면 ─────────────
# 주소에 종목이 있으면(?code=...) 차트 화면, 없으면 리스트 화면
qp = st.query_params
KEEP = {k: qp.get(k) for k in ("embed", "embed_options") if qp.get(k)}   # 감싼 화면(index.html)용 주소 옵션 유지


def link(**kw) -> str:
    return "?" + urlencode({**KEEP, **kw})


def n2(v):
    return "-" if v is None else f"{v:,.2f}"


def money(v):
    """거래대금($) → '123억$' (1억 달러 미만은 '0.5억$')"""
    if v is None:
        return "-"
    e = v / 1e8
    return f"{e:,.0f}억$" if e >= 10 else f"{e:,.1f}억$"


def ud(v, base):     # 전일 종가보다 높으면 빨강(up), 낮으면 파랑(dn)
    if v is None or base is None:
        return "eq"
    return "up" if v > base else ("dn" if v < base else "eq")


LIST_CSS = """<style>
.sl-top{display:flex;align-items:center;gap:8px;margin:4px 0 8px 0}
.sl-top a.tab{display:inline-block;min-width:92px;text-align:center;padding:8px 14px;font-size:16px;font-weight:700;
 border:1px solid #c9ced6;background:#f4f5f7;color:#555 !important;border-radius:8px;text-decoration:none !important}
.sl-top a.tab.on{background:#1f4fa8;border-color:#1f4fa8;color:#fff !important}
.sl-top a.tab.rf{min-width:0;font-size:14px;font-weight:600;color:#1f4fa8 !important;background:#fff;border-color:#9db4e0}
.sl-top .asof{margin-left:auto;font-size:12px;color:#888;white-space:nowrap}
.tm{font-size:12px;color:#888;white-space:nowrap}.tm b{font-weight:600}
.sl-top .brk{display:none}
@media (max-width:600px){.sl-top{flex-wrap:wrap;row-gap:6px}.sl-top .brk{display:block;flex-basis:100%;height:0}.sl-top .tm{flex:1 1 auto;font-size:12px}.sl-top .asof{font-size:12px}}
.sl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid #c9ced6}
.sl-wrap table{border-collapse:collapse;width:100%;min-width:820px;font-size:13px;margin:0}
.sl-wrap th{background:#e9edf3;color:#333;font-weight:600;padding:6px;border:1px solid #d3d8df;white-space:nowrap;text-align:center}
.sl-wrap td{padding:0;border:1px solid #e3e6ea;white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
.sl-wrap td a{display:block;padding:6px;color:inherit !important;text-decoration:none !important}
.sl-wrap td.rk{text-align:center;color:#888;width:34px}
.sl-wrap td.nm{text-align:left;font-weight:700;color:#111;max-width:210px;overflow:hidden;text-overflow:ellipsis}
.sl-wrap td.nm .sub{display:block;font-size:11px;font-weight:400;color:#999;overflow:hidden;text-overflow:ellipsis}
.sl-wrap td.tv{color:#333}
.sl-wrap tr:nth-child(even) td{background:#fafbfc}
.sl-wrap tbody tr:hover td{background:#eef4ff}
.sl-wrap td.sg{text-align:center;font-weight:800;min-width:76px}.sl-wrap td.sg.buy{color:#e5484d}.sl-wrap td.sg.sell{color:#1a56db}.sl-wrap td.sg.wait{color:#bbb;font-weight:400}
.sl-wrap .up{color:#e5484d}.sl-wrap .dn{color:#1a56db}.sl-wrap .eq{color:#333}
@media (max-width:600px){
 .sl-top a.tab{min-width:80px;padding:7px 10px;font-size:15px}
 .sl-top a.tab.rf{min-width:0;padding:7px 8px;font-size:13px}
 .sl-wrap table{min-width:680px}
 .sl-wrap td.nm{max-width:118px}
 .sl-wrap th.rk,.sl-wrap td.rk{position:sticky;left:0;z-index:2;background:#fff}
 .sl-wrap th.nm,.sl-wrap td.nm{position:sticky;left:34px;z-index:2;background:#fff;box-shadow:1px 0 0 #d3d8df}
 .sl-wrap th.rk,.sl-wrap th.nm{background:#e9edf3}
 .sl-wrap tr:nth-child(even) td.rk,.sl-wrap tr:nth-child(even) td.nm{background:#fafbfc}
}
</style>"""


if not qp.get("code"):
    if qp.get("refresh"):                                         # ↻ 새로고침을 눌렀음 → 시세 다시 받기
        top_list.clear()
        idx_hist.clear()
        with _store()["lock"]:
            _store()["data"].clear()
        del st.query_params["refresh"]
    with st.spinner("미국 주식 거래대금 순위를 불러오는 중입니다…"):
        _tl = time.time()
        rows, asof = top_list()
        t_list = time.time() - _tl
    if not rows:
        st.error("종목 리스트를 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
        st.stop()

    def build(sig, tm=""):
        h = [LIST_CSS, '<div class="sl-top">']
        h.append(f'<a class="tab on" href="{link()}" target="_self">미국 거래대금 TOP{TOP_N}</a>')
        h.append(f'<a class="tab rf" href="{link(refresh=1)}" target="_self" title="시세 다시 받기">↻ 새로고침</a><span class="brk"></span>')
        h.append(tm or '<span class="tm">⏱ 측정 중…</span>')
        h.append(f'<span class="asof">기준 {asof} (한국시간)</span></div>')
        h.append('<div class="sl-wrap"><table><thead><tr><th class="rk">순위</th>'
                 '<th class="nm">종목</th><th>신호</th><th>현재가($)</th><th>대비</th><th>등락률</th>'
                 '<th>거래대금</th><th>시가</th><th>고가</th><th>저가</th></tr></thead><tbody>')
        for i, (c, nm, close, chg, pct, op, hi, lo, tv, ex) in enumerate(rows, 1):
            href = html.escape(link(code=c, name=nm, ex=ex))
            a = f'<a href="{href}" target="_self">'
            prev = close - chg if (close is not None and chg is not None) else None
            k = "eq" if pct is None else ("up" if pct > 0 else ("dn" if pct < 0 else "eq"))
            arrow = "" if not pct else ("▲ " if pct > 0 else "▼ ")
            pct_t = "-" if pct is None else f"{pct:+.2f}%" if pct else "0.00%"
            nm_e, c_e = html.escape(nm), html.escape(c)
            sub = f'<span class="sub">{nm_e}</span>' if nm and nm.upper() != c.upper() else ""
            if sig is None or c not in sig:                   # 아직 계산 중
                sg = '<td class="sg wait">' + a + '…</a></td>'
            else:
                v = sig.get(c)
                sg = ('<td class="sg buy">' + a + '매수 가능</a></td>' if v == "buy" else
                      '<td class="sg sell">' + a + '매도 가능</a></td>' if v == "sell" else
                      '<td class="sg">' + a + '&nbsp;</a></td>')
            h.append(
                f'<tr><td class="rk">{a}{i}</a></td>'
                f'<td class="nm" title="{nm_e}">{a}{c_e}{sub}</a></td>'
                f'{sg}'
                f'<td class="{k}">{a}{n2(close)}</a></td>'
                f'<td class="{k}">{a}{arrow}{n2(None if chg is None else abs(chg))}</a></td>'
                f'<td class="{k}">{a}{pct_t}</a></td>'
                f'<td class="tv">{a}{money(tv)}</a></td>'
                f'<td class="{ud(op, prev)}">{a}{n2(op)}</a></td>'
                f'<td class="{ud(hi, prev)}">{a}{n2(hi)}</a></td>'
                f'<td class="{ud(lo, prev)}">{a}{n2(lo)}</a></td></tr>')
        h.append('</tbody></table></div>')
        return "".join(h)                             # 줄바꿈 없이 한 덩어리로 넣어야 표가 깨지지 않음

    codes = tuple(r[0] for r in rows)
    days = max(int(calc), 180)
    note = st.empty()
    ph = st.empty()
    known = known_signals(codes, days)                        # 보관된 신호(받기·계산 없이 바로)
    ph.markdown(build(known), unsafe_allow_html=True)         # ① 리스트를 항상 먼저 표시
    if len(known) < len(codes):
        note.caption("⏳ 매수·매도 신호를 계산하는 중입니다… (처음 한 번은 조금 걸려요)")
    _t0 = time.time()
    try:
        sig, _got, t_fetch, t_calc = list_signals(codes)
    except Exception as e:
        sig, _got, t_fetch, t_calc = {}, len(codes), time.time() - _t0, 0.0
        st.caption(f"신호 계산 오류: {e}")
    note.empty()
    if _got < len(rows):          # 시세를 다 못 받았으면 알려줌 (빈칸 = 계산 못 한 종목 포함)
        st.caption(f"※ {len(rows)}종목 중 {_got}종목만 시세를 받아 신호를 계산했습니다 ({time.time() - _t0:.0f}초).")
    _tm = timing_html([("리스트", t_list), ("시세", t_fetch), ("계산", t_calc)])
    ph.markdown(build(sig, _tm), unsafe_allow_html=True)      # ② 신호 칸 + 걸린 시간 채워서 표시
    st.stop()


# ───────────── ② 차트 화면 (주식001 차트) ─────────────
_t_chart0 = time.time()
BACK_CSS = ('<style>.cv-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:2px 0 4px 0}'
            '.cv-top a.back{display:inline-block;padding:4px 14px;font-size:15px;font-weight:600;border:1px solid #c9ced6;'
            'border-radius:6px;background:#f4f5f7;color:#333 !important;text-decoration:none !important}'
            '.tm{font-size:12px;color:#888;white-space:nowrap}.tm b{font-weight:600}</style>')


def top_bar(tm: str) -> str:
    """← 목록으로 버튼 + 걸린 시간"""
    return (BACK_CSS + '<div class="cv-top">'
            f'<a class="back" href="{html.escape(link())}" target="_self">← 목록으로</a>{tm}</div>')


top_ph = st.empty()
top_ph.markdown(top_bar('<span class="tm">⏱ 측정 중…</span>'), unsafe_allow_html=True)

code = yticker(qp.get("code"))
name = str(qp.get("name") or code)
ex = str(qp.get("ex") or "").upper()
name_ph = st.empty()
name_ph.markdown(
    f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
    f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>',
    unsafe_allow_html=True)

# 시세를 받는 동안 "고장난 것처럼" 보이지 않도록 안내 표시 (차트가 나오면 사라짐)
status = st.empty()
if cached_hist(code, max(int(calc), 180)) is None:
    status.markdown(
        '<div style="padding:10px 12px;border-radius:8px;background:#fff4dc;color:#8a5a00;font-size:14px;line-height:1.5">'
        '⏳ 시세를 불러오는 중입니다. 처음 한 번은 20~30초 걸릴 수 있어요. 잠시만 기다려 주세요.</div>',
        unsafe_allow_html=True)

def load_both(code: str, idx_name: str, days: int):
    """종목과 지수 시세를 동시에 받아 온다. 지수는 8초 안에 못 받으면 None(차트는 지수선 없이 표시)"""
    ex = ThreadPoolExecutor(max_workers=2)
    f_stock = ex.submit(hist, code, int(days))
    f_idx = ex.submit(idx_hist, idx_name, int(days))
    try:
        stock = f_stock.result(timeout=40)
        try:
            idx = f_idx.result(timeout=8)
        except Exception:
            idx = None
        return stock, idx
    finally:
        ex.shutdown(wait=False)


calc = max(int(calc), 180)
idx_name = "나스닥" if ex == "NASDAQ" else "S&P500"   # 나스닥 상장 → 나스닥지수, 그 외 → S&P500
try:
    _tf = time.time()
    df, _idf = load_both(code, idx_name, calc)
    t_fetch = time.time() - _tf
except Exception as e:
    status.empty()
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()
if df.empty or len(df) < 60:
    status.empty()
    st.warning("데이터가 부족합니다(60개 이상 필요). 티커를 확인하세요.")
    st.stop()

_tc = time.time()
y = df["Close"].to_numpy(dtype=float)
n = len(y)
t = np.arange(n)

# 곡선추세·매도선·매수선 (리스트 신호와 같은 계산)
curve, upper, lower = calc_bands(y)

# 화면에는 최근 disp일만 표시
D = n
init_span = int(min(disp, D))
sl = slice(n - D, n)
dates = df.index.strftime("%Y-%m-%d")[sl]
yv, curvev, upv, lov = y[sl], curve[sl], upper[sl], lower[sl]
def _ohl(c):   # 시가·고가·저가가 비어 있으면 종가로 대신 채움
    a = df[c].to_numpy(dtype=float)[sl] if c in df.columns else yv.copy()
    return np.where(np.isfinite(a) & (a > 0), a, yv)
ov, hv, lv = _ohl("Open"), _ohl("High"), _ohl("Low")
hv, lv = np.maximum.reduce([hv, ov, yv]), np.minimum.reduce([lv, ov, yv])

# ───────────── 지수(나스닥/S&P500) 비교선 ─────────────
idx_arr = None
_live_ok = False
try:
    _s = _idf["Close"].reindex(df.index).ffill().bfill()
    if len(_s) == n and np.isfinite(_s.to_numpy(dtype=float)).all():
        idx_arr = _s.to_numpy(dtype=float)
except Exception:
    idx_arr = None
idx_note = ""
if idx_arr is not None:     # 지수 데이터가 종목보다 며칠 일찍 끝나면 알려줌 (마지막 값으로 채워졌다는 뜻)
    _gap = (df.index.max() - _idf.index.max()).days
    if _gap >= 1 and not _live_ok:
        idx_note = f"※ {idx_name} 데이터가 {_idf.index.max():%Y-%m-%d}까지만 있어 최근 {_gap}일은 마지막 값으로 채웠습니다."
idx_scaled = None
if idx_arr is not None:     # 처음 열었을 때 맨 왼쪽 종가에 한 번만 맞춘 뒤 고정 (밀거나 확대해도 변하지 않음)
    idx_scaled = idx_arr * (yv[n - init_span] / idx_arr[n - init_span])
PHONE_SPAN_PY = min(42, init_span)
idx_scaled_p = None
if idx_scaled is not None:  # 폰은 처음 2달치만 보이므로 그 맨 왼쪽 종가에 맞춘 지수선도 따로 준비
    idx_scaled_p = idx_arr * (yv[n - PHONE_SPAN_PY] / idx_arr[n - PHONE_SPAN_PY])

# ───────────── 상단 시세 헤더 ─────────────
last, prev = y[-1], y[-2]
chg = last - prev
pct = chg / prev * 100
col = "#e5484d" if chg > 0 else ("#3b6fd4" if chg < 0 else "#333333")
arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "-")
vol = df["Volume"].iloc[-1]

# ───────────── 종목명 옆: 기준 날짜 + 지수 ─────────────
if idx_arr is not None:
    name_ph.markdown(
        f'<div style="display:flex;align-items:baseline;gap:18px;flex-wrap:wrap">'
        f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
        f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>'
        f'<div style="font-size:17px;font-weight:700;color:#0aa05a">'
        f'● {df.index[-1]:%Y-%m-%d} {idx_name} {idx_arr[-1]:,.2f}</div>'
        '</div>',
        unsafe_allow_html=True)

# ───────────── 매수·매도 표시 (현재가가 곡선추세 상·하한에 닿거나 넘었을 때) ─────────────
_up, _lo = float(upper[-1]), float(lower[-1])
sig_html = ""
if last >= _up * (1 - TOUCH_TOL):          # 상한에 닿거나 위로 넘음 → 파란색 매도
    sig_html = '<div style="font-size:40px;font-weight:800;color:#1a56db;line-height:1">매도</div>'
elif last <= _lo * (1 + TOUCH_TOL):        # 하한에 닿거나 아래로 내려감 → 빨간색 매수
    sig_html = '<div style="font-size:40px;font-weight:800;color:#e5484d;line-height:1">매수</div>'

_sig = ""
if sig_html:
    _sig_txt, _sig_col = ("매도 가능합니다", "#1a56db") if "매도" in sig_html else ("매수 가능합니다", "#e5484d")
    _sig = f'<div class="hd-sig" style="color:{_sig_col}">{_sig_txt}</div>'
st.markdown(
    '<style>'
    '.hd-row{display:flex;align-items:center;gap:16px;flex-wrap:nowrap}'
    '.hd-px{display:flex;flex-direction:column;align-items:flex-start;gap:4px}'
    '.hd-big{font-size:42px;font-weight:700;line-height:1}'
    '.hd-chg{font-size:16px}'
    '.hd-lines{display:flex;flex-direction:column;gap:10px}'
    '.hd-lbl{font-size:15px;color:#111;line-height:1.4;text-align:center;white-space:nowrap}'
    '.hd-sig{font-size:24px;font-weight:800;line-height:1.2;white-space:nowrap}'
    '@media (max-width:600px){'
    '.hd-row{gap:10px}.hd-big{font-size:30px}.hd-chg{font-size:14px}'
    '.hd-lbl{font-size:12px}.hd-sig{font-size:17px}}'
    '</style>'
    '<div class="hd-row">'
    '<div>'
    f'<div class="hd-px"><span class="hd-big" style="color:{col}">${last:,.2f}</span>'
    f'<span class="hd-chg" style="color:{col}">{arrow} {abs(chg):,.2f} &nbsp; {pct:+.2f}%</span></div>'
    f'<div style="font-size:13px;color:#888;margin-top:6px">거래량 {vol:,.0f}</div>'
    '</div>'
    '<div class="hd-lines">'
    f'<div class="hd-lbl" data-ul="1" style="color:#1a56db;font-weight:600">매도선<br>(<span data-k="u">{_up:,.2f}</span>)</div>'
    f'<div class="hd-lbl" data-ul="1" style="color:#e5484d;font-weight:600">매수선<br>(<span data-k="l">{_lo:,.2f}</span>)</div>'
    '</div>'
    f'{_sig}'
    '</div>',
    unsafe_allow_html=True)

# ───────────── 차트 ─────────────
def direction_colors(v):
    d = np.diff(v, prepend=v[1])
    d[0] = d[1]
    return np.where(d >= 0, "#e5484d", "#3b6fd4")  # 상승=빨강, 하락=파랑


def base_traces(fig):
    fig.add_trace(go.Candlestick(x=dates, open=ov, high=hv, low=lv, close=yv, name="수정종가", showlegend=False,
                                 increasing=dict(line=dict(color="#e5484d", width=1), fillcolor="#e5484d"),
                                 decreasing=dict(line=dict(color="#3b6fd4", width=1), fillcolor="#3b6fd4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=curvev, name="곡선추세", line=dict(color="#111111", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=upv, name="매도선", line=dict(color="#1a56db", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lov, name="매수선", line=dict(color="#e5484d", width=2.2)), row=1, col=1)
    if idx_scaled is not None:   # 4번 트레이스: 지수(녹색 점선)
        fig.add_trace(go.Scatter(x=dates, y=idx_scaled, name=idx_name,
                                 line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)
        if idx_scaled_p is not None:   # 5번 트레이스: 폰용 지수선(처음엔 숨김, 폰이면 JS가 이것으로 교체)
            fig.add_trace(go.Scatter(x=dates, y=idx_scaled_p, name=idx_name, visible=False, showlegend=False,
                                     line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)


CROSSHAIR_JS = """
function installCrosshair(gd, initSpan, S, B, IDX, IT, IDXNAME, IDXRAW, OPEN, LOW, ANCH, IDX2, IT2, HIGH){
  var LP_MS=500;           // 길게 눌러 진동이 오기까지의 시간(ms)
  var CROSS_DELAY_MS=150;  // 진동 후 십자선이 나타나기까지의 시간(ms) (진동 → 십자선 순서 맞춤)
  var PAN_GAIN=1.5;        // 손가락 이동 속도 배율
  var ZOOM_GAIN=1.5;       // 확대/축소 속도 배율
  var COMPARE_MODE='level'; // 'level'=그날 종목 가격과 그날 지수선(녹색 점선) 비교, 'daily'=그날 등락률끼리 비교
  var SAME_BAND_LEVEL=2;    // level: 가격 차이가 ±2% 이내면 '보합'
  var SAME_BAND_DAILY=0.3;  // daily: 등락률 차이가 ±0.3%p 이내면 '보합'
  var idxOn=true;          // 지수 비교선 표시 여부
  var RPAD_RATIO=0.08;     // 오른쪽 여백: 보이는 기간의 8%만큼 빈 칸 (120일 → 약 10칸, 폰 42일 → 약 3칸)
  var RPAD_MIN=3;          // 오른쪽 여백 최소 칸 수
  // 브라우저(앱 브라우저)가 두 손가락으로 화면 전체를 확대하지 못하게 막기
  try{
    var pd=window.parent.document, mv=pd.querySelector('meta[name=viewport]');
    if(mv)mv.setAttribute('content','width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');
    if(!pd.getElementById('nozoom_css')){
      var st=pd.createElement('style'); st.id='nozoom_css';
      st.textContent='iframe{touch-action:pan-y !important;}'; pd.head.appendChild(st);
    }
  }catch(err){}
  gd.style.position='relative'; gd.style.touchAction='pan-y';
  gd.style.userSelect='none'; gd.style.webkitUserSelect='none';
  gd.style.webkitTouchCallout='none'; gd.style.webkitTapHighlightColor='transparent';
  var cats=gd.data[0].x, N=cats.length, MINS=Math.min(20,N), MAXS=Math.min(Math.max(120,initSpan),N);
  function clampSpan(v){return Math.round(Math.max(MINS,Math.min(MAXS,v)));}
  function clampI0(v,sp){return Math.round(Math.max(0,Math.min(N-sp,v)));}
  // 핸드폰(화면 폭 600px 이하)은 처음에 약 2달(42거래일)만 확대해서 보여줌. PC는 그대로.
  var PHONE_SPAN=42, ITa=IT;
  var isPhone=false;
  try{ isPhone=Math.min(window.screen&&window.screen.width||9999, window.innerWidth||9999)<=600; }catch(e0){}
  if(isPhone && initSpan>PHONE_SPAN && IDX && IDX2 && IDX2.length===N){
    try{   // 폰: 2달치 기준으로 미리 환산해 둔 지수선으로 교체(보이기/숨기기만 사용)
      IDX=IDX2; ITa=IT2; initSpan=PHONE_SPAN;
      var pr0=Plotly.restyle(gd,{visible:[false]},[IT]); if(pr0&&pr0.then)pr0.then(null,function(){});
      var pr1=Plotly.restyle(gd,{visible:[true]},[IT2]); if(pr1&&pr1.then)pr1.then(null,function(){});
    }catch(e1){}
  } else if(isPhone && initSpan>PHONE_SPAN){ initSpan=PHONE_SPAN; }
  var span=clampSpan(initSpan), i0=N-span;
  var pending=false, mode=null, timer=null, sx=0, sy=0, startI0=0, lastT=null, mdown=false, pinch=null, markersOn=null;
  var tStart=0, lastTap=0, lastTouch=0;

  function mk(css){var d=document.createElement('div');
    d.style.cssText='position:absolute;pointer-events:none;display:none;z-index:50;'+css;
    gd.appendChild(d);return d;}
  var lab='background:#222;color:#fff;font:11px sans-serif;padding:2px 4px;white-space:nowrap;';
  var vl=mk('width:0;border-left:1px solid #333;');
  var hl=mk('height:0;border-top:1px solid #333;');
  var dl=mk(lab+'transform:translateX(-50%);white-space:pre;text-align:center;line-height:1.3;');   // 날짜 + (가격 , 지수) 두 줄
  var pr=mk(lab+'width:50px;transform:translateY(-50%);');
  function f2(v){return Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
  var sm=mk('left:4px;top:6px;font:600 12px sans-serif;white-space:nowrap;');   // 지수 대비 요약
  var dt=mk('left:4px;top:24px;font:11px sans-serif;color:#666;white-space:nowrap;');   // 계산 근거(기준일부터의 수익률)
  var ig=mk('left:4px;top:40px;font:600 11px sans-serif;color:#00a651;white-space:nowrap;');   // 지수 값(녹색)
  if(IDX){sm.style.display='block'; dt.style.display='block'; ig.style.display='block';}
  var focus=N-1;           // 십자선이 가리키는 날짜(없으면 맨 오른쪽=최근)
  function sg(v){return (v>=0?'+':'')+v.toFixed(1)+'%';}
  function md(i){return cats[i].slice(5);}
  function pxs(v){return Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
  function updateSummary(){   // 선택한 날짜(없으면 최근 날짜)의 종목 가격과 지수로 강함/약함/보합 판단
    var D0=focus;
    try{
      var su=window.parent.document.querySelectorAll('[data-k=u]'), sl2=window.parent.document.querySelectorAll('[data-k=l]');
      for(var q=0;q<su.length;q++)su[q].textContent=pxs(S[2][D0]);
      for(var q2=0;q2<sl2.length;q2++)sl2[q2].textContent=pxs(S[3][D0]);}catch(e2){}
    if(!IDX)return;
    var R=IDXRAW||IDX, D=focus, d, band, row2, unit;
    if(COMPARE_MODE==='daily' && D>=1){
      var sr=(S[0][D]/S[0][D-1]-1)*100, ir=(R[D]/R[D-1]-1)*100;
      d=sr-ir; band=SAME_BAND_DAILY; unit='%p ';
      row2='그날 등락 종목 '+sg(sr)+' · '+IDXNAME+' '+sg(ir);
    } else {
      d=(S[0][D]/IDX[D]-1)*100; band=SAME_BAND_LEVEL; unit='% ';     // 종목 가격이 지수선보다 몇 % 위/아래인지
      // 시가·저가·종가는 전일 종가 대비, 지수선은 전일 지수선 대비 등락률
      var prevClose = D > 0 ? S[0][D-1] : NaN;
      var prevIndex = D > 0 ? IDX[D-1] : NaN;
      function coloredValue(label, value, base, pctSourceValue){
        // pctSourceValue가 있으면 실제 원본 지수값으로 등락률을 계산
        var compareValue = (pctSourceValue !== undefined) ? pctSourceValue : value;
        var pct = (isFinite(base) && base !== 0 && isFinite(compareValue))
                    ? (compareValue / base - 1) * 100 : NaN;
        var color = !isFinite(pct) || Math.abs(pct) < 0.0000001 ? '#000000' : (pct > 0 ? '#e5484d' : '#3b6fd4');
        var pctText = isFinite(pct) ? (pct > 0 ? '+' : '') + pct.toFixed(2) + '%' : '—';
        return '<span style="color:'+color+'">'+label+' '+pxs(value)+' ('+pctText+')</span>';
      }
      var prevRawIndex = D > 0 ? R[D-1] : NaN;
      row2 = '<span style="color:#000000">시가 '+pxs(OPEN[D])+'</span> · ' +
       '<span style="color:#000000">저가 '+pxs(LOW[D])+'</span> · ' +
       '<span style="color:#000000">종가 '+pxs(S[0][D])+'</span> · ' +
       coloredValue('지수', R[D], prevRawIndex);
    }
    var up=d>band, dn=d<-band;
    sm.textContent=IDXNAME+' 대비 '+(d>=0?'+':'')+d.toFixed(1)+unit+(up?'강함':(dn?'약함':'보합'));
    sm.style.color=up?'#e5484d':(dn?'#3b6fd4':'#666');
    if(COMPARE_MODE==='daily'){
      dt.textContent=row2;
    } else {
      dt.innerHTML=row2;
    }
    ig.textContent='● '+cats[D]+'  '+IDXNAME+' '+f2(R[D]);
  }
  var tg=null;
  var dbg=mk('left:4px;top:66px;font:11px sans-serif;color:#c00;background:rgba(255,255,255,.92);padding:2px 4px;max-width:92%;z-index:70;');
  function showErr(e){dbg.style.display='block'; dbg.textContent='차트 오류: '+(e&&e.message?e.message:e);}
  function hide(){[vl,hl,dl,pr].forEach(function(e){e.style.display='none';}); if(focus!==N-1){focus=N-1; updateSummary();}}

  function fmt(c){return c.slice(5).replace('-','.');}

  // 보이는 구간의 최고/최저에 맞춰 가격 범위를 즉시 변경 + 지수선을 맨 왼쪽 종가에 맞춰 환산
  function draw(){
    var a=Math.max(0,Math.floor(i0)), b=Math.min(N-1,Math.ceil(i0+span)-1), i, k, v;
    var lo=Infinity, hi=-Infinity, bl=0, bh=0;
    for(k=0;k<S.length;k++){for(i=a;i<=b;i++){v=S[k][i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(IDX&&idxOn){for(i=a;i<=b;i++){v=IDX[i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(HIGH&&LOW){for(i=a;i<=b;i++){if(HIGH[i]>hi)hi=HIGH[i]; if(LOW[i]<lo)lo=LOW[i];}}   // 캔들 고가·저가
    for(i=a;i<=b;i++){v=B[i]; if(v<bl)bl=v; if(v>bh)bh=v;}
    if(!isFinite(lo)||!isFinite(hi)||!isFinite(bl)||!isFinite(bh))throw new Error('가격 범위 계산 오류');
    var pad=(hi-lo)*0.06||1;
    var step=Math.max(3,Math.round(span/4)), tv=[], tt=[], idx;
    for(idx=N-1;idx>=a;idx-=step){tv.push(cats[idx]); tt.push(fmt(cats[idx]));}
    var rp=Math.max(RPAD_MIN,Math.round(span*RPAD_RATIO));   // 오늘 캔들이 오른쪽 끝에 붙지 않게 빈 칸 추가
    var xr=[i0-0.5,i0+span-0.5+rp];
    var lay={'xaxis.range':xr,'xaxis2.range':xr,
      'yaxis.range':[lo-pad,hi+pad],'yaxis2.range':[bl*1.1,(bh*1.1)||1],
      'xaxis.tickvals':tv,'xaxis.ticktext':tt,'xaxis2.tickvals':tv,'xaxis2.ticktext':tt};
    var p=Plotly.relayout(gd,lay);
    if(p&&p.then)p.then(null,showErr);
  }
  function render(){
    pending=false;
    try{ draw(); }
    catch(e){                                // 지수 부분에서 문제가 생겨도 차트는 계속 그리도록 지수 기능만 끔
      showErr(e);
      if(IDX){
        IDX=null; sm.style.display='none'; if(tg)tg.style.display='none';
        try{ Plotly.restyle(gd,{visible:[false]},[ITa]); }catch(e1){}
        try{ draw(); }catch(e2){ showErr(e2); }
      }
    }
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}

  function move(cx,cy){
    var r=gd.getBoundingClientRect(), x=cx-r.left, y=cy-r.top;
    var fl=gd._fullLayout, s=fl._size, xa=fl.xaxis, ya=fl.yaxis, yb=fl.yaxis2;
    if(x<s.l||x>s.l+s.w||y<s.t||y>s.t+s.h){hide();return;}
    var lo=xa.range[0], hi=xa.range[1];
    var idx=Math.round((x-xa._offset)/xa._length*(hi-lo)+lo);
    idx=Math.max(0,Math.min(N-1,idx));
    focus=idx; updateSummary();
    var xs=xa._offset+(idx-lo)/(hi-lo)*xa._length;
    if(!isFinite(xs))return;
    var ax=(y>=yb._offset)?yb:ya;
    var v=ax.range[1]-(y-ax._offset)/ax._length*(ax.range[1]-ax.range[0]);
    var txt=Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    vl.style.cssText+=';display:block;left:'+xs+'px;top:'+s.t+'px;height:'+s.h+'px;';
    hl.style.cssText+=';display:block;left:'+s.l+'px;top:'+y+'px;width:'+s.w+'px;';
    var showI=!!(IDX&&IDXRAW&&idxOn);
    // 예) 2026-10-20 (줄바꿈) (11,700 , 5,600.00)  → 날짜 / (그날 종목 가격 , 그날 지수)
    dl.textContent=cats[idx]+'\\n('+pxs(S[0][idx])+(showI?' , '+f2(IDXRAW[idx]):'')+')';
    dl.style.display='block'; dl.style.top=(ya._offset+ya._length+2)+'px';
    var half=(dl.offsetWidth||0)/2, tot=fl.width||gd.clientWidth||0;
    dl.style.left=(tot?Math.max(half,Math.min(tot-half,xs)):xs)+'px';        // 화면 밖으로 나가지 않게
    pr.style.display='block'; pr.style.left=(s.l+s.w+2)+'px'; pr.style.top=y+'px'; pr.textContent=txt;
  }
  function pxToBars(dx){return dx/gd._fullLayout._size.w*span;}
  function fracOf(clientX){
    var r=gd.getBoundingClientRect(), s=gd._fullLayout._size;
    return Math.max(0,Math.min(1,(clientX-r.left-s.l)/s.w));
  }
  function dist(t){var dx=t[0].clientX-t[1].clientX, dy=t[0].clientY-t[1].clientY; return Math.sqrt(dx*dx+dy*dy);}
  function resetView(){                       // 더블클릭/더블탭: 처음 상태로
    span=clampSpan(initSpan); i0=N-span; pinch=null; if(mode==='cross')mode=null; hide(); schedule();
  }
  function zoomBy(factor){                    // 화면 가운데 기준 확대/축소
    var c=i0+span/2, ns=clampSpan(span*factor);
    if(ns===span)ns=clampSpan(span+(factor<1?-1:1));
    span=ns; i0=clampI0(c-ns/2,ns); hide(); schedule();
  }

  // ── 확대/축소 버튼 (두 손가락 확대가 막히는 폰용) ──
  function btn(txt,right,fn,w,fs){
    var b=document.createElement('div'); b.textContent=txt;
    b.style.cssText='position:absolute;top:2px;right:'+right+'px;width:'+(w||28)+'px;height:22px;line-height:22px;text-align:center;'
      +'font:600 '+(fs||15)+'px sans-serif;background:#f3f3f3;border:1px solid #ccc;border-radius:4px;z-index:60;cursor:pointer;'
      +'touch-action:manipulation;user-select:none;-webkit-user-select:none;';
    ['touchstart','touchmove','touchend','mousedown','mousemove','dblclick'].forEach(function(n){
      b.addEventListener(n,function(e){e.stopPropagation();},{passive:true});});
    b.addEventListener('click',function(e){e.stopPropagation();fn();});
    gd.appendChild(b);
    return b;
  }
  btn('＋',92,function(){zoomBy(0.6);});
  btn('－',62,function(){zoomBy(1/0.6);});
  if(IDX){                                  // 지수 비교선 켜기/끄기 버튼 (버튼 글자 = 지수 이름)
    tg=btn(IDXNAME,124,function(){idxOn=!idxOn; paintTg(); try{Plotly.restyle(gd,{visible:[idxOn]},[ITa]);}catch(e){showErr(e);} schedule();},46,12);
    var paintTg=function(){tg.style.color=idxOn?'#007a3d':'#999'; tg.style.background=idxOn?'#e8f8ef':'#f3f3f3'; tg.style.borderColor=idxOn?'#00a651':'#ccc';};
    paintTg();
  }

  // ── 손가락 ──
  // 바로 좌우로 밀기=기간 이동(한 칸씩 딱딱하게) / 길게 눌러 진동=십자선 / 두 손가락=확대·축소
  gd.addEventListener('touchstart',function(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches.length>=2){
      hide(); mode='pinch';
      pinch={d:dist(e.touches),span:span,anchor:i0+fracOf((e.touches[0].clientX+e.touches[1].clientX)/2)*span};
      return;
    }
    hide();
    var t=e.touches[0]; lastT=t; sx=t.clientX; sy=t.clientY; startI0=i0; mode=null; tStart=Date.now();
    timer=setTimeout(function(){
      if(mode!==null)return;
      mode='arming'; if(navigator.vibrate)navigator.vibrate(40);       // ① 진동
      timer=setTimeout(function(){                                      // ② 잠깐 뒤 십자선 표시 후 따라다님
        if(mode==='arming'){mode='cross'; move(lastT.clientX,lastT.clientY);}
      },CROSS_DELAY_MS);
    },LP_MS);
  },{passive:true});
  gd.addEventListener('touchmove',function(e){
    if(mode==='pinch'){
      if(e.touches.length===2&&pinch){
        e.preventDefault();
        var f=fracOf((e.touches[0].clientX+e.touches[1].clientX)/2);
        var ns=clampSpan(pinch.span*Math.pow(pinch.d/dist(e.touches),ZOOM_GAIN));
        span=ns; i0=clampI0(pinch.anchor-f*ns,ns); schedule();
      }
      return;
    }
    if(e.touches.length!==1)return;
    var t=e.touches[0]; lastT=t;
    if(mode==='arming'){e.preventDefault(); return;}
    if(mode==='cross'){e.preventDefault(); move(t.clientX,t.clientY); return;}
    if(mode==='pan'){e.preventDefault(); i0=clampI0(startI0-pxToBars(t.clientX-sx)*PAN_GAIN,span); schedule(); return;}
    if(mode==='scroll')return;
    var dx=t.clientX-sx, dy=t.clientY-sy;
    if(Math.abs(dx)>6||Math.abs(dy)>6){
      clearTimeout(timer);
      if(Math.abs(dx)>Math.abs(dy)){mode='pan'; e.preventDefault(); i0=clampI0(startI0-pxToBars(dx)*PAN_GAIN,span); schedule();}
      else mode='scroll';
    }
  },{passive:false});
  function touchEnd(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches&&e.touches.length>0)return;      // 아직 손가락이 남아 있음
    pinch=null;
    if(mode===null&&e.type==='touchend'&&Date.now()-tStart<300){   // 짧게 톡 → 두 번이면 기본값
      var now=Date.now();
      if(now-lastTap<350){resetView(); lastTap=0;} else lastTap=now;
    }
    if(mode!=='cross')mode=null;                   // 십자선은 손을 떼도 남고, 다음 터치 때 사라짐
  }
  gd.addEventListener('touchend',touchEnd);
  gd.addEventListener('touchcancel',touchEnd);
  gd.addEventListener('contextmenu',function(e){e.preventDefault();});

  // ── PC: 마우스를 누른 채 끌면 이동, 그냥 움직이면 십자선, Ctrl+휠=확대/축소 ──
  function emu(e){return (Date.now()-lastTouch<1000)||(e.sourceCapabilities&&e.sourceCapabilities.firesTouchEvents);}
  gd.addEventListener('mousedown',function(e){if(emu(e))return; mdown=true; sx=e.clientX; startI0=i0; e.preventDefault();});
  window.addEventListener('mousemove',function(e){
    if(!mdown)return;
    i0=clampI0(startI0-pxToBars(e.clientX-sx),span); schedule();
  },true);
  window.addEventListener('mouseup',function(){mdown=false;},true);
  gd.addEventListener('mousemove',function(e){if(emu(e))return; move(e.clientX,e.clientY);});
  gd.addEventListener('mouseleave',function(e){if(emu(e))return; hide();});
  gd.addEventListener('dblclick',function(e){resetView();});
  gd.addEventListener('wheel',function(e){
    if(!e.ctrlKey)return; e.preventDefault();
    var f=fracOf(e.clientX), anchor=i0+f*span;
    var ns=clampSpan(span*(e.deltaY>0?1.23:0.81));
    if(ns===span)ns=clampSpan(span+(e.deltaY>0?1:-1));
    span=ns; i0=clampI0(anchor-f*ns,ns); schedule();
  },{passive:false});
  updateSummary();
  render();
}
"""


def finish(fig, low_label):
    m = len(dates)
    pos = sorted({m - 1 - int(round(i * (m - 1) / 3.6)) for i in range(4)})
    fig.update_xaxes(type="category", tickmode="array",
                     tickvals=[dates[i] for i in pos],
                     ticktext=[pd.to_datetime(dates[i]).strftime("%m.%d") for i in pos],
                     tickangle=0, tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#f0b8b8", gridwidth=1,
                     showline=False, zeroline=False)
    fig.update_xaxes(showticklabels=True, row=1, col=1)    # 날짜는 위 차트 아래(두 차트 사이)에 표시
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(side="right", tickformat=",.2f", tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#eeeeee", zeroline=False, showline=False, ticks="")
    fig.update_yaxes(nticks=4, row=2, col=1)
    tagc = "#e5484d" if chg > 0 else "#3b6fd4"
    # 현재가 태그(오른쪽 축)
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", yref="y", y=yv[-1], showarrow=False,
                       text=f"{last:,.2f}<br>{pct:+.2f}%", align="left", borderpad=2,
                       font=dict(color="white", size=11), bgcolor=tagc)
    # 범례(차트 위 한 줄)
    fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                       showarrow=False, font=dict(size=11),
                       text=('<span style="color:#e5484d">▮</span><span style="color:#3b6fd4">▮</span> <span style="color:#111111">수정종가 캔들</span>  '
                             '<span style="color:#111111">━ 곡선</span>  '
                             '<span style="color:#1a56db">━ 상한</span>  '
                             '<span style="color:#e5484d">━ 하한</span>'))
    # 아래 차트 이름
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=fig.layout.yaxis2.domain[1],
                       xanchor="left", yanchor="top", showarrow=False,
                       text=low_label, font=dict(size=11, color="#555"))
    fig.update_layout(height=620, margin=dict(l=4, r=66, t=(78 if idx_arr is not None else 26), b=22),
                      showlegend=False, bargap=0.05, hovermode=False, dragmode=False,
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show(fig, div_id, bars):
    """좌우 이동 + 길게 눌러 십자선 기능을 얹어 차트를 표시"""
    def pack(v):
        return np.round(np.asarray(v, dtype=float), 2).tolist()
    series = [pack(v) for v in (yv, curvev, upv, lov)]
    dump = lambda o: json.dumps(o, separators=(",", ":"))
    idx_json = dump(pack(idx_scaled)) if idx_scaled is not None else "null"
    raw_json = dump(pack(idx_arr)) if idx_arr is not None else "null"
    open_json = dump(pack(ov))
    low_json = dump(pack(lv))
    idx2_json = dump(pack(idx_scaled_p)) if idx_scaled_p is not None else "null"
    post = "installCrosshair(document.getElementById('{plot_id}'), %d, %s, %s, %s, %d, %s, %s, %s, %s, %d, %s, %d, %s);" % (
        init_span, dump(series), dump(pack(bars)),
        idx_json, (4 if idx_scaled is not None else -1), json.dumps(idx_name),
        raw_json, open_json, low_json, n - init_span, idx2_json, 5, dump(pack(hv)))
    html = "<script>" + CROSSHAIR_JS + "</script>" + fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id=div_id,
        config={"staticPlot": True, "responsive": True},
        post_script=post)
    components.html(html, height=640)


t_calc = time.time() - _tc
_td = time.time()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
base_traces(fig)
cv = curvev - curvev.min()
fig.add_trace(go.Bar(x=dates, y=cv, marker_color=direction_colors(curvev),
                     name="곡선추세 방향", showlegend=False), row=2, col=1)
show(finish(fig, "곡선추세 방향"), "chart_trend", cv)
status.empty()
top_ph.markdown(top_bar(timing_html([("시세", t_fetch), ("계산", t_calc), ("차트", time.time() - _td)])),
                unsafe_allow_html=True)
if idx_arr is None:
    st.caption("※ 지수 데이터를 불러오지 못해 비교선 없이 표시 중입니다.")
elif idx_note:
    st.caption(idx_note)
