"""
KR Signal Backend — pykrx 없이 네이버 금융 직접 호출
"""
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
import requests, traceback, os

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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.naver.com",
}

def fetch_ohlcv(code, pages=6):
    rows = []
    for page in range(1, pages + 1):
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            tbls = pd.read_html(r.text)
            for tbl in tbls:
                tbl = tbl.dropna(how="all")
                if tbl.shape[1] < 7:
                    continue
                tbl.columns = ["date","close","diff","open","high","low","volume"]
                tbl = tbl[tbl["date"].astype(str).str.match(r"\d{4}\.\d{2}\.\d{2}")]
                if len(tbl) == 0:
                    continue
                for col in ["close","open","high","low","volume"]:
                    tbl[col] = pd.to_numeric(
                        tbl[col].astype(str).str.replace(",",""), errors="coerce")
                tbl["date"] = pd.to_datetime(tbl["date"])
                rows.append(tbl[["date","open","high","low","close","volume"]])
        except Exception as e:
            print(f"[WARN] {code} page {page}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[df["close"] > 0].copy()

def calc_rsi(closes, period=14):
    delta = closes.diff()
    ag = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bb(closes, period=20, mult=2):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std

def calc_macd(closes, fast=12, slow=26, sig=9):
    m = closes.ewm(span=fast,adjust=False).mean() - closes.ewm(span=slow,adjust=False).mean()
    return m, m.ewm(span=sig, adjust=False).mean()

def detect_signals(df):
    signals = []
    if len(df) < 30:
        return signals
    rsi  = df["rsi"].iloc[-1]
    bbu  = df["bbu"].iloc[-1]
    bbm  = df["bbm"].iloc[-1]
    bbl  = df["bbl"].iloc[-1]
    macd = df["macd"].iloc[-1]; macd_p = df["macd"].iloc[-2]
    sig  = df["sig"].iloc[-1];  sig_p  = df["sig"].iloc[-2]
    c=df["close"].iloc[-1]; cp=df["close"].iloc[-2]
    o=df["open"].iloc[-1];  op=df["open"].iloc[-2]
    lp=df["low"].iloc[-2];  hp=df["high"].iloc[-2]

    if rsi <= 30:
        if lp < bbl and (c>op) and (o<cp) and (c>o):
            signals.append({"type":"A","direction":"BUY","level":2,
                "title":"강력 매수 신호 [전략A]",
                "desc":f"RSI {rsi:.1f} 과매도+BB하단 이탈+장악형 양봉",
                "stop":round(bbl*0.99),"target":round(bbm+(bbm-bbl))})
        elif lp < bbl:
            signals.append({"type":"A","direction":"WATCH","level":1,
                "title":"관심 등록 [전략A]",
                "desc":f"RSI {rsi:.1f} 과매도+BB이탈. 장악형 캔들 대기.",
                "stop":None,"target":None})

    if rsi >= 70 and hp > bbu and (c<op) and (o>cp) and (c<o):
        signals.append({"type":"A","direction":"SELL","level":2,
            "title":"강력 매도 신호 [전략A]",
            "desc":f"RSI {rsi:.1f} 과매수+BB상단+장악형 음봉",
            "stop":round(bbu*1.01),"target":round(bbm-(bbu-bbm))})

    if rsi <= 35 and len(df) >= 6:
        c4=df["close"].iloc[-5]; r4=df["rsi"].iloc[-5]
        cross=(macd_p<sig_p)and(macd>sig)
        bull=(c>op)and(o<cp)
        if (c<c4)and(rsi>r4)and cross and bull:
            sv=float(df["low"].iloc[-5:].min())*0.99
            signals.append({"type":"B","direction":"BUY","level":2,
                "title":"추세 반전 확정 [전략B]",
                "desc":"상승 다이버전스+MACD 골든크로스+장악형 3중 확인",
                "stop":round(sv),"target":round(c+(c-sv)*2)})
        elif (c<c4)and(rsi>r4):
            signals.append({"type":"B","direction":"WATCH","level":1,
                "title":"추세 전환 징후 [전략B]",
                "desc":"상승 다이버전스 감지. MACD+장악형 캔들 대기.",
                "stop":None,"target":None})
    return signals

def build_stock_data(code):
    df = fetch_ohlcv(code)
    if len(df) < 30:
        return None
    df["rsi"] = calc_rsi(df["close"])
    df["bbu"], df["bbm"], df["bbl"] = calc_bb(df["close"])
    df["macd"], df["sig"] = calc_macd(df["close"])
    df = df.dropna()
    if len(df) < 10:
        return None
    signals = detect_signals(df)
    wins, total = 0, 0
    for i in range(20, len(df)-5):
        if df["rsi"].iloc[i] <= 30:
            total += 1
            if df["close"].iloc[i+5] > df["close"].iloc[i]:
                wins += 1
    last = df.iloc[-1]; prev = df.iloc[-2]
    chg  = ((last["close"]-prev["close"])/prev["close"])*100
    return {
        "price":    int(last["close"]),
        "change":   round(float(chg),2),
        "rsi":      round(float(df["rsi"].iloc[-1]),1),
        "macdVal":  round(float(df["macd"].iloc[-1]),0),
        "sigVal":   round(float(df["sig"].iloc[-1]),0),
        "bbUpper":  int(last["bbu"]),
        "bbMid":    int(last["bbm"]),
        "bbLower":  int(last["bbl"]),
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
            traceback.print_exc()
    return jsonify(result)

@app.route("/api/stock/<code>")
def get_stock(code):
    s = next((x for x in STOCKS if x["code"]==code), None)
    if not s:
        return jsonify({"error":"not found"}),404
    try:
        data = build_stock_data(code)
        return jsonify({**s,**data}) if data else (jsonify({"error":"no data"}),503)
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","source":"naver-finance","time":datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",5001)))
