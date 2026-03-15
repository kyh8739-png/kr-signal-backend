"""
KR Signal Backend — pykrx 기반 (나중에 KIS API로 교체 가능)
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pykrx import stock
import traceback, os

app = Flask(__name__)
CORS(app)

# ─── 대상 종목 ──────────────────────────────────────────────────────
STOCKS = [
    {"code": "005930", "name": "삼성전자",      "sector": "반도체",    "cap": 4200},
    {"code": "000660", "name": "SK하이닉스",    "sector": "반도체",    "cap": 1100},
    {"code": "035420", "name": "NAVER",          "sector": "IT",        "cap": 380},
    {"code": "035720", "name": "카카오",          "sector": "IT",        "cap": 220},
    {"code": "005380", "name": "현대차",          "sector": "자동차",    "cap": 580},
    {"code": "051910", "name": "LG화학",          "sector": "2차전지",   "cap": 340},
    {"code": "006400", "name": "삼성SDI",         "sector": "2차전지",   "cap": 290},
    {"code": "247540", "name": "에코프로비엠",    "sector": "2차전지",   "cap": 180},
    {"code": "068270", "name": "셀트리온",        "sector": "바이오",    "cap": 260},
    {"code": "207940", "name": "삼성바이오",      "sector": "바이오",    "cap": 520},
    {"code": "373220", "name": "LG에너지솔루션", "sector": "2차전지",   "cap": 850},
    {"code": "000270", "name": "기아",            "sector": "자동차",    "cap": 480},
]

# ─── 기술 지표 계산 ────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bb(closes, period=20, mult=2):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + mult * std, mid, mid - mult * std

def calc_macd(closes, fast=12, slow=26, sig=9):
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal

# ─── 신호 감지 ─────────────────────────────────────────────────────
def detect_signals(df):
    signals = []
    if len(df) < 30:
        return signals

    rsi   = df["rsi"].iloc[-1]
    rsi_p = df["rsi"].iloc[-2]
    bb_upper = df["bb_upper"].iloc[-1]
    bb_lower = df["bb_lower"].iloc[-1]
    bb_mid   = df["bb_mid"].iloc[-1]
    macd     = df["macd"].iloc[-1]
    macd_p   = df["macd"].iloc[-2]
    sig      = df["signal"].iloc[-1]
    sig_p    = df["signal"].iloc[-2]
    close    = df["close"].iloc[-1]
    close_p  = df["close"].iloc[-2]
    open_    = df["open"].iloc[-1]
    open_p   = df["open"].iloc[-2]
    low_p    = df["low"].iloc[-2]
    high_p   = df["high"].iloc[-2]

    # ── 전략 A: 과매도 ──
    if rsi <= 30:
        bb_break = low_p < bb_lower
        bull_engulf = (close > open_p) and (open_ < close_p) and (close > open_)
        if bb_break and bull_engulf:
            signals.append({
                "type": "A", "direction": "BUY", "level": 2,
                "title": "강력 매수 신호 [전략A]",
                "desc": f"RSI {rsi:.1f} 과매도 + BB하단 이탈 + 장악형 양봉",
                "stop": round(bb_lower * 0.99),
                "target": round(bb_mid + (bb_mid - bb_lower)),
            })
        elif bb_break:
            signals.append({
                "type": "A", "direction": "WATCH", "level": 1,
                "title": "관심 등록 [전략A]",
                "desc": f"RSI {rsi:.1f} 과매도 + BB하단 이탈. 장악형 캔들 대기.",
                "stop": None, "target": None,
            })

    # ── 전략 A: 과매수 ──
    if rsi >= 70:
        bb_break = high_p > bb_upper
        bear_engulf = (close < open_p) and (open_ > close_p) and (close < open_)
        if bb_break and bear_engulf:
            signals.append({
                "type": "A", "direction": "SELL", "level": 2,
                "title": "강력 매도 신호 [전략A]",
                "desc": f"RSI {rsi:.1f} 과매수 + BB상단 이탈 + 장악형 음봉",
                "stop": round(bb_upper * 1.01),
                "target": round(bb_mid - (bb_upper - bb_mid)),
            })

    # ── 전략 B: 다이버전스 ──
    if rsi <= 35 and len(df) >= 6:
        close_4ago = df["close"].iloc[-5]
        rsi_4ago   = df["rsi"].iloc[-5]
        price_lower  = close < close_4ago
        rsi_higher   = rsi > rsi_4ago
        macd_cross   = (macd_p < sig_p) and (macd > sig)
        bull_engulf  = (close > open_p) and (open_ < close_p)
        if price_lower and rsi_higher and macd_cross and bull_engulf:
            stop_val = float(df["low"].iloc[-5:].min()) * 0.99
            signals.append({
                "type": "B", "direction": "BUY", "level": 2,
                "title": "추세 반전 확정 [전략B]",
                "desc": "상승 다이버전스 + MACD 골든크로스 + 장악형 양봉 3중 확인",
                "stop": round(stop_val),
                "target": round(close + (close - stop_val) * 2),
            })
        elif price_lower and rsi_higher:
            signals.append({
                "type": "B", "direction": "WATCH", "level": 1,
                "title": "추세 전환 징후 [전략B]",
                "desc": "상승 다이버전스 감지. MACD + 장악형 캔들 대기.",
                "stop": None, "target": None,
            })

    return signals

# ─── pykrx 데이터 로드 ─────────────────────────────────────────────
def fetch_ohlcv(code, days=120):
    end   = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start, end, code)
    df.columns = ["open", "high", "low", "close", "volume", "trading_value", "price_change_rate"]
    df = df[df["close"] > 0].copy()
    return df

def build_stock_data(code):
    df = fetch_ohlcv(code, days=120)
    if len(df) < 30:
        return None

    df["rsi"]      = calc_rsi(df["close"])
    bb_u, bb_m, bb_l = calc_bb(df["close"])
    df["bb_upper"] = bb_u
    df["bb_mid"]   = bb_m
    df["bb_lower"] = bb_l
    df["macd"], df["signal"] = calc_macd(df["close"])
    df = df.dropna()

    signals = detect_signals(df)

    # 백테스트: RSI≤30 신호 후 5봉 수익률
    wins, total = 0, 0
    for i in range(20, len(df) - 5):
        if df["rsi"].iloc[i] <= 30:
            total += 1
            if df["close"].iloc[i + 5] > df["close"].iloc[i]:
                wins += 1

    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = ((last["close"] - prev["close"]) / prev["close"]) * 100

    # 스파크라인 (최근 30봉 종가)
    sparkline = df["close"].iloc[-30:].round().tolist()

    return {
        "price":    int(last["close"]),
        "change":   round(float(change), 2),
        "rsi":      round(float(df["rsi"].iloc[-1]), 1),
        "macdVal":  round(float(df["macd"].iloc[-1]), 0),
        "sigVal":   round(float(df["signal"].iloc[-1]), 0),
        "bbUpper":  int(last["bb_upper"]),
        "bbMid":    int(last["bb_mid"]),
        "bbLower":  int(last["bb_lower"]),
        "volume":   int(last["volume"]),
        "signals":  signals,
        "winRate":  round(wins / total * 100) if total > 0 else 0,
        "winTotal": total,
        "sparkline": sparkline,
        # 백테스트용 히스토리 (최근 60봉)
        "history": [
            {
                "date":   str(d.date()),
                "open":   int(r["open"]),
                "high":   int(r["high"]),
                "low":    int(r["low"]),
                "close":  int(r["close"]),
                "volume": int(r["volume"]),
                "rsi":    round(float(r["rsi"]), 1) if not np.isnan(r["rsi"]) else None,
                "bbU":    int(r["bb_upper"]) if not np.isnan(r["bb_upper"]) else None,
                "bbM":    int(r["bb_mid"])   if not np.isnan(r["bb_mid"])   else None,
                "bbL":    int(r["bb_lower"]) if not np.isnan(r["bb_lower"]) else None,
                "macd":   round(float(r["macd"]), 0) if not np.isnan(r["macd"]) else None,
                "sig":    round(float(r["signal"]), 0) if not np.isnan(r["signal"]) else None,
            }
            for d, r in df.iloc[-60:].iterrows()
        ],
    }

# ─── API 엔드포인트 ────────────────────────────────────────────────
@app.route("/api/stocks")
def get_stocks():
    result = []
    for s in STOCKS:
        try:
            data = build_stock_data(s["code"])
            if data:
                result.append({**s, **data})
        except Exception as e:
            print(f"[ERROR] {s['code']}: {e}")
            traceback.print_exc()
    return jsonify(result)

@app.route("/api/stock/<code>")
def get_stock(code):
    s = next((x for x in STOCKS if x["code"] == code), None)
    if not s:
        return jsonify({"error": "not found"}), 404
    try:
        data = build_stock_data(code)
        if not data:
            return jsonify({"error": "data unavailable"}), 503
        return jsonify({**s, **data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest/<code>")
def backtest(code):
    """
    특정 종목 전략A/B 백테스트 — 과거 신호 전체 추출
    """
    s = next((x for x in STOCKS if x["code"] == code), None)
    if not s:
        return jsonify({"error": "not found"}), 404
    try:
        df = fetch_ohlcv(code, days=365)
        df["rsi"]            = calc_rsi(df["close"])
        bb_u, bb_m, bb_l     = calc_bb(df["close"])
        df["bb_upper"]       = bb_u
        df["bb_mid"]         = bb_m
        df["bb_lower"]       = bb_l
        df["macd"], df["signal"] = calc_macd(df["close"])
        df = df.dropna()

        trades = []
        for i in range(30, len(df) - 10):
            window = df.iloc[:i+1]
            sigs   = detect_signals(window)
            for sig in sigs:
                if sig["level"] == 2 and sig["direction"] in ("BUY", "SELL"):
                    entry = int(df["close"].iloc[i])
                    exit5 = int(df["close"].iloc[i + 5])
                    exit10= int(df["close"].iloc[i + 10])
                    pnl5  = round((exit5  - entry) / entry * 100, 2)
                    pnl10 = round((exit10 - entry) / entry * 100, 2)
                    if sig["direction"] == "SELL":
                        pnl5, pnl10 = -pnl5, -pnl10
                    trades.append({
                        "date":     str(df.index[i].date()),
                        "type":     sig["type"],
                        "direction":sig["direction"],
                        "entry":    entry,
                        "exit5":    exit5,
                        "exit10":   exit10,
                        "pnl5":     pnl5,
                        "pnl10":    pnl10,
                        "win5":     pnl5 > 0,
                        "rsi":      round(float(df["rsi"].iloc[i]), 1),
                    })

        wins5  = sum(1 for t in trades if t["win5"])
        total  = len(trades)
        avg5   = round(sum(t["pnl5"] for t in trades) / total, 2) if total else 0
        avg10  = round(sum(t["pnl10"] for t in trades) / total, 2) if total else 0

        return jsonify({
            "stock":    s,
            "trades":   trades[-30:],  # 최근 30건
            "summary": {
                "total":    total,
                "wins":     wins5,
                "winRate":  round(wins5 / total * 100) if total else 0,
                "avgPnl5":  avg5,
                "avgPnl10": avg10,
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "source": "pykrx", "time": datetime.now().isoformat()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
