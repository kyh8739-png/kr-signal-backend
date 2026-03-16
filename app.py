"""
KR Signal Backend — KIS API + 네이버 금융 폴백
신호: Level1(관찰) / Level2(강력)
백테스트: 손절선(BB하단/전저점) + 손익비(RR) 포함
"""
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests, traceback, os

app = Flask(__name__)
CORS(app)

KIS_APP_KEY    = os.environ.get("KIS_APP_KEY", "")
KIS_APP_SECRET = os.environ.get("KIS_APP_SECRET", "")
KIS_ACCOUNT    = os.environ.get("KIS_ACCOUNT", "")
KIS_MODE       = os.environ.get("KIS_MODE", "virtual")
BASE_URL = ("https://openapivts.koreainvestment.com:29443" if KIS_MODE == "virtual"
            else "https://openapi.koreainvestment.com:9443")
_token_cache = {"token": None, "expires": None}

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

def get_token():
    now = datetime.now()
    if _token_cache["token"] and _token_cache["expires"] and now < _token_cache["expires"]:
        return _token_cache["token"]
    r = requests.post(f"{BASE_URL}/oauth2/tokenP", json={
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET,
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    _token_cache["token"]   = data["access_token"]
    _token_cache["expires"] = now + timedelta(hours=23)
    return _token_cache["token"]

def kis_headers(tr_id):
    return {"Content-Type": "application/json",
            "authorization": f"Bearer {get_token()}",
            "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id, "custtype": "P"}

def fetch_ohlcv_kis(code):
    end   = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")
    tr_id = "VTTC8416R" if KIS_MODE == "virtual" else "FHKST03010100"
    url   = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
              "FID_INPUT_DATE_1": start, "FID_INPUT_DATE_2": end,
              "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0"}
    r = requests.get(url, headers=kis_headers(tr_id), params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("rt_cd") != "0":
        raise Exception(f"KIS: {data.get('msg1')}")
    rows = []
    for item in data.get("output2", []):
        try:
            rows.append({"date": pd.to_datetime(item["stck_bsop_date"]),
                         "open": float(item["stck_oprc"]), "high": float(item["stck_hgpr"]),
                         "low":  float(item["stck_lwpr"]), "close": float(item["stck_clpr"]),
                         "volume": float(item["acml_vol"])})
        except: continue
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df[df["close"] > 0].copy()

def fetch_ohlcv_naver(code, pages=25):
    HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com"}
    rows = []
    for page in range(1, pages + 1):
        try:
            r    = requests.get(f"https://finance.naver.com/item/sise_day.naver?code={code}&page={page}",
                                headers=HEADERS, timeout=10)
            tbls = pd.read_html(r.text)
            for tbl in tbls:
                tbl = tbl.dropna(how="all")
                if tbl.shape[1] < 7: continue
                tbl.columns = ["date","close","diff","open","high","low","volume"]
                tbl = tbl[tbl["date"].astype(str).str.match(r"\d{4}\.\d{2}\.\d{2}")]
                if len(tbl) == 0: continue
                for col in ["close","open","high","low","volume"]:
                    tbl[col] = pd.to_numeric(tbl[col].astype(str).str.replace(",",""), errors="coerce")
                tbl["date"] = pd.to_datetime(tbl["date"])
                rows.append(tbl[["date","open","high","low","close","volume"]])
        except Exception as e:
            print(f"[WARN] naver {code} p{page}: {e}")
    if not rows: return pd.DataFrame()
    df = pd.concat(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[df["close"] > 0].copy()

def fetch_ohlcv(code):
    if KIS_APP_KEY and KIS_APP_SECRET:
        try:
            df = fetch_ohlcv_kis(code)
            if len(df) >= 30: return df, "kis"
        except Exception as e:
            print(f"[WARN] KIS {code}: {e}")
    return fetch_ohlcv_naver(code), "naver"

def calc_rsi(closes, period=14):
    delta = closes.diff()
    ag = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def calc_bb(closes, period=20, mult=2):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std

def calc_macd(closes, fast=12, slow=26, sig=9):
    m = closes.ewm(span=fast,adjust=False).mean() - closes.ewm(span=slow,adjust=False).mean()
    return m, m.ewm(span=sig, adjust=False).mean()

def calc_stop(df, i):
    """손절선: BB하단 vs 최근 5봉 저점 중 낮은 값 * 0.99"""
    bb_l  = float(df["bbl"].iloc[i])
    low5  = float(df["low"].iloc[max(0, i-4):i+1].min())
    return round(min(bb_l, low5) * 0.99)

def detect_signals(df):
    signals = []
    if len(df) < 6: return signals
    rsi    = df["rsi"].iloc[-1]
    macd   = df["macd"].iloc[-1]; macd_p = df["macd"].iloc[-2]
    sig    = df["sig"].iloc[-1];  sig_p  = df["sig"].iloc[-2]
    c      = df["close"].iloc[-1]
    c4     = df["close"].iloc[-5]
    r4     = df["rsi"].iloc[-5]
    rsi_ok   = rsi <= 35
    div_ok   = (c < c4) and (rsi > r4)
    cross_ok = (macd_p < sig_p) and (macd > sig)

    if rsi_ok and div_ok and cross_ok:
        stop = calc_stop(df, -1)
        risk = c - stop
        signals.append({"type":"2","direction":"BUY","level":2,
            "title":"강력 진입 신호",
            "desc":f"RSI {rsi:.1f} + 상승 다이버전스 + MACD 골든크로스",
            "stop":   stop,
            "target": round(c + risk * 2),   # 1:2
            "target15": round(c + risk * 1.5) # 1:1.5
        })
    elif rsi_ok and div_ok:
        stop = calc_stop(df, -1)
        risk = c - stop
        signals.append({"type":"1","direction":"WATCH","level":1,
            "title":"관찰 신호",
            "desc":f"RSI {rsi:.1f} + 상승 다이버전스. MACD 골든크로스 대기.",
            "stop":   stop,
            "target": round(c + risk * 2),
            "target15": round(c + risk * 1.5)
        })
    return signals

def build_stock_data(code):
    df, source = fetch_ohlcv(code)
    if len(df) < 30: return None
    df["rsi"]              = calc_rsi(df["close"])
    df["bbu"],df["bbm"],df["bbl"] = calc_bb(df["close"])
    df["macd"],df["sig"]   = calc_macd(df["close"])
    df = df.dropna()
    if len(df) < 10: return None
    signals = detect_signals(df)
    wins, total = 0, 0
    for i in range(5, len(df)-5):
        rsi_i = df["rsi"].iloc[i]; c_i = df["close"].iloc[i]
        c_i4  = df["close"].iloc[i-4]; rsi_i4 = df["rsi"].iloc[i-4]
        if rsi_i <= 35 and (c_i < c_i4) and (rsi_i > rsi_i4):
            total += 1
            if df["close"].iloc[i+5] > c_i: wins += 1
    last = df.iloc[-1]; prev = df.iloc[-2]
    chg  = ((last["close"]-prev["close"])/prev["close"])*100
    return {
        "price":    int(last["close"]),  "change":   round(float(chg),2),
        "rsi":      round(float(df["rsi"].iloc[-1]),1),
        "macdVal":  round(float(df["macd"].iloc[-1]),0),
        "sigVal":   round(float(df["sig"].iloc[-1]),0),
        "bbUpper":  int(last["bbu"]),    "bbMid":    int(last["bbm"]),
        "bbLower":  int(last["bbl"]),    "volume":   int(last["volume"]),
        "signals":  signals,
        "winRate":  round(wins/total*100) if total>0 else 0,
        "winTotal": total,
        "sparkline": df["close"].iloc[-30:].round().tolist(),
        "source":   source,
    }

@app.route("/api/stocks")
def get_stocks():
    def fetch_one(s):
        try:
            data = build_stock_data(s["code"])
            return {**s, **data} if data else None
        except Exception as e:
            print(f"[ERROR] {s['code']}: {e}"); return None
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_one, s): s for s in STOCKS}
        result  = [f.result() for f in as_completed(futures) if f.result()]
    order = [s["code"] for s in STOCKS]
    result.sort(key=lambda x: order.index(x["code"]) if x["code"] in order else 99)
    return jsonify(result)

@app.route("/api/stock/<code>")
def get_stock(code):
    s = next((x for x in STOCKS if x["code"]==code), None)
    if not s: return jsonify({"error":"not found"}),404
    try:
        data = build_stock_data(code)
        return jsonify({**s,**data}) if data else (jsonify({"error":"no data"}),503)
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/backtest/<code>")
def backtest(code):
    s = next((x for x in STOCKS if x["code"]==code), None)
    if not s: return jsonify({"error":"not found"}),404
    try:
        df, src = fetch_ohlcv(code)
        if len(df) < 30: return jsonify({"error":"데이터 부족"}),503
        df["rsi"]              = calc_rsi(df["close"])
        df["bbu"],df["bbm"],df["bbl"] = calc_bb(df["close"])
        df["macd"],df["sig"]   = calc_macd(df["close"])
        df = df.dropna()

        trades = []
        for i in range(5, len(df)-10):
            rsi_i  = df["rsi"].iloc[i];  c_i    = df["close"].iloc[i]
            c_i4   = df["close"].iloc[i-4]; rsi_i4 = df["rsi"].iloc[i-4]
            macd_i = df["macd"].iloc[i];  macd_p = df["macd"].iloc[i-1]
            sig_i  = df["sig"].iloc[i];   sig_p  = df["sig"].iloc[i-1]
            div_ok   = (c_i < c_i4) and (rsi_i > rsi_i4)
            cross_ok = (macd_p < sig_p) and (macd_i > sig_i)
            if not (rsi_i <= 35 and div_ok): continue

            level = 2 if cross_ok else 1
            entry = int(c_i)
            exit5 = int(df["close"].iloc[i+5])
            exit10= int(df["close"].iloc[i+10])
            pnl5  = round((exit5-entry)/entry*100, 2)
            pnl10 = round((exit10-entry)/entry*100, 2)

            # 손절선: BB하단 vs 전저점 중 낮은 값
            bb_l  = float(df["bbl"].iloc[i])
            low5  = float(df["low"].iloc[max(0,i-4):i+1].min())
            stop  = round(min(bb_l, low5) * 0.99)
            risk  = entry - stop if entry > stop else 1
            target_2r  = round(entry + risk * 2)
            target_15r = round(entry + risk * 1.5)
            rr_actual  = round((exit5 - entry) / risk, 2) if risk > 0 else 0

            idx = df.index[i]
            date_str = str(idx.date()) if hasattr(idx,"date") else str(df["date"].iloc[i])[:10]

            trades.append({
                "date":      date_str,
                "level":     level,
                "direction": "BUY",
                "entry":     entry,
                "stop":      stop,
                "target2r":  target_2r,
                "target15r": target_15r,
                "exit5":     exit5,
                "exit10":    exit10,
                "pnl5":      pnl5,
                "pnl10":     pnl10,
                "win5":      pnl5 > 0,
                "rsi":       round(float(rsi_i), 1),
                "rr_actual": rr_actual,
            })

        wins  = sum(1 for t in trades if t["win5"])
        total = len(trades)
        avg5  = round(sum(t["pnl5"]  for t in trades)/total, 2) if total else 0
        avg10 = round(sum(t["pnl10"] for t in trades)/total, 2) if total else 0
        return jsonify({"stock":s,"trades":trades[-30:],
            "summary":{"total":total,"wins":wins,
                "winRate":round(wins/total*100) if total else 0,
                "avgPnl5":avg5,"avgPnl10":avg10}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","mode":KIS_MODE,
        "source":"kis" if KIS_APP_KEY else "naver",
        "kis_ready":bool(KIS_APP_KEY),"time":datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",5001)))
