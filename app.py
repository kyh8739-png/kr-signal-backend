# pkg_resources 호환성 패치 (Python 3.12 + pykrx)
import sys, types
_pkg = types.ModuleType("pkg_resources")
_pkg.declare_namespace = lambda *a, **k: None
_pkg.require = lambda *a, **k: None
_pkg.get_distribution = lambda *a, **k: None
_pkg.DistributionNotFound = Exception
_pkg.VersionConflict = Exception
sys.modules.setdefault("pkg_resources", _pkg)

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback, os

# pykrx import (패치 후)
from pykrx import stock

app = Flask(__name__)
CORS(app)

STOCKS = [
    {"code": "005930", "name": "삼성전자",       "sector": "반도체",   "cap": 4200},
    {"code": "000660", "name": "SK하이닉스",     "sector": "반도체",   "cap": 1100},
    {"code": "035420", "name": "NAVER",           "sector": "IT",       "cap": 380 },
    {"code": "035720", "name": "카카오",           "sector": "IT",       "cap": 220 },
    {"code": "005380", "name": "현대차",           "sector": "자동차",   "cap": 580 },
    {"code": "051910", "name": "LG화학",           "sector": "2차전지",  "cap": 340 },
    {"code": "006400", "name": "삼성SDI",          "sector": "2차전지",  "cap": 290 },
    {"code": "247540", "name": "에코프로비엠",     "sector": "2차전지",  "cap": 180 },
    {"code": "068270", "name": "셀트리온",         "sector": "바이오",   "cap": 260 },
    {"code": "207940", "name": "삼성바이오",       "sector": "바이오",   "cap": 520 },
    {"code": "373220", "name": "LG에너지솔루션",  "sector": "2차전지",  "cap": 850 },
    {"code": "000270", "name": "기아",             "sector": "자동차",   "cap": 480 },
]

def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period-1, min_periods=period).mean()
    al = loss.ewm(com=period-1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bb(closes, period=20, mult=2):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std

def calc_macd(closes, fast=12, slow=26, sig=9):
    ef = closes.ewm(span=fast, adjust=False).mean()
    es = closes.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=sig, adjust=False).mean()
    return m, s

def detect_signals(df):
    signals = []
    if len(df) < 30:
        return signals
    rsi    = df["rsi"].iloc[-1]
    bb_u   = df["bb_upper"].iloc[-1]
    bb_m   = df["bb_mid"].iloc[-1]
    bb_l   = df["bb_lower"].iloc[-1]
    macd   = df["macd"].iloc[-1]
    macd_p = df["macd"].iloc[-2]
    sig    = df["signal"].iloc[-1]
    sig_p  = df["signal"].iloc[-2]
    c      = df["close"].iloc[-1]
    c_p    = df["close"].iloc[-2]
    o      = df["open"].iloc[-1]
    o_p    = df["open"].iloc[-2]
    l_p    = df["low"].iloc[-2]
    h_p    = df["high"].iloc[-2]

    if rsi <= 30:
        bb_break   = l_p < bb_l
        bull       = (c > o_p) and (o < c_p) and (c > o)
        if bb_break and bull:
            signals.append({"type":"A","direction":"BUY","level":2,
                "title":"강력 매수 신호 [전략A]",
                "desc":f"RSI {rsi:.1f} 과매도+BB하단 이탈+장악형 양봉",
                "stop":round(bb_l*0.99),"target":round(bb_m+(bb_m-bb_l))})
        elif bb_break:
            signals.append({"type":"A","direction":"WATCH","level":1,
                "title":"관심 등록 [전략A]",
                "desc":f"RSI {rsi:.1f} 과매도+BB이탈. 장악형 캔들 대기.",
                "stop":None,"target":None})

    if rsi >= 70:
        bb_break = h_p > bb_u
        bear     = (c < o_p) and (o > c_p) and (c < o)
        if bb_break and bear:
            signals.append({"type":"A","direction":"SELL","level":2,
                "title":"강력 매도 신호 [전략A]",
                "desc":f"RSI {rsi:.1f} 과매수+BB상단 이탈+장악형 음봉",
                "stop":round(bb_u*1.01),"target":round(bb_m-(bb_u-bb_m))})

    if rsi <= 35 and len(df) >= 6:
        c4     = df["close"].iloc[-5]
        r4     = df["rsi"].iloc[-5]
        pl     = c < c4
        rh     = rsi > r4
        cross  = (macd_p < sig_p) and (macd > sig)
        bull   = (c > o_p) and (o < c_p)
        if pl and rh and cross and bull:
            stop_v = float(df["low"].iloc[-5:].min())*0.99
            signals.append({"type":"B","direction":"BUY","level":2,
                "title":"추세 반전 확정 [전략B]",
                "desc":"상승 다이버전스+MACD 골든크로스+장악형 3중 확인",
                "stop":round(stop_v),"target":round(c+(c-stop_v)*2)})
        elif pl and rh:
            signals.append({"type":"B","direction":"WATCH","level":1,
                "title":"추세 전환 징후 [전략B]",
                "desc":"상승 다이버전스 감지. MACD+장악형 캔들 대기.",
                "stop":None,"target":None})
    return signals

def fetch_ohlcv(code, days=120):
    end   = datetime.today().strftime("%Y%m%d")
    start = (datetime.today()-timedelta(days=days)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start, end, code)
    df.columns = ["open","high","low","close","volume","trading_value","price_change_rate"]
    return df[df["close"]>0].copy()

def build_stock_data(code):
    df = fetch_ohlcv(code)
    if len(df) < 30:
        return None
    df["rsi"]              = calc_rsi(df["close"])
    bu, bm, bl             = calc_bb(df["close"])
    df["bb_upper"]         = bu
    df["bb_mid"]           = bm
    df["bb_lower"]         = bl
    df["macd"], df["signal"] = calc_macd(df["close"])
    df = df.dropna()
    signals = detect_signals(df)
    wins, total = 0, 0
    for i in range(20, len(df)-5):
        if df["rsi"].iloc[i] <= 30:
            total += 1
            if df["close"].iloc[i+5] > df["close"].iloc[i]:
                wins += 1
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    chg   = ((last["close"]-prev["close"])/prev["close"])*100
    return {
        "price":    int(last["close"]),
        "change":   round(float(chg),2),
        "rsi":      round(float(df["rsi"].iloc[-1]),1),
        "macdVal":  round(float(df["macd"].iloc[-1]),0),
        "sigVal":   round(float(df["signal"].iloc[-1]),0),
        "bbUpper":  int(last["bb_upper"]),
        "bbMid":    int(last["bb_mid"]),
        "bbLower":  int(last["bb_lower"]),
        "volume":   int(last["volume"]),
        "signals":  signals,
        "winRate":  round(wins/total*100) if total>0 else 0,
        "winTotal": total,
        "sparkline": df["close"].iloc[-30:].round().tolist(),
    }

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
    return jsonify(result)

@app.route("/api/stock/<code>")
def get_stock(code):
    s = next((x for x in STOCKS if x["code"]==code), None)
    if not s:
        return jsonify({"error":"not found"}), 404
    try:
        data = build_stock_data(code)
        return jsonify({**s, **data}) if data else jsonify({"error":"no data"}), 503
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","source":"pykrx","time":datetime.now().isoformat()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
