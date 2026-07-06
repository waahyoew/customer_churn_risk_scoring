# ⚡ Quick Start Guide — Churn Prediction API

Get up and running in **5 minutes**.

---

## 📋 Prerequisites

- Python 3.11+ installed
- Dataset file: `data/cth_churn_analysis_train.xlsx`

---

## 🚀 Step-by-Step

### 1. Install Dependencies (30 seconds)

```bash
cd e:\ML_AI_engineer\datascientist\Project\churn_prediction_pipeline
pip install -r requirements.txt
```

---

### 2. Train Model (1 minute)

```bash
python train.py
```

**Expected output:**
```
Training completed successfully!
Model saved to: models/churn_model_v2.joblib
Metadata saved to: models/churn_model_v2_metadata.json
```

---

### 3. Start API Server (10 seconds)

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

### 4. Test API (30 seconds)

#### Option A: Interactive Swagger UI

Open in browser: **http://localhost:8000/docs**

1. Click on `/predict` → **Try it out**
2. Use example payload (auto-populated)
3. Click **Execute**

#### Option B: curl Command

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"Jenis_kelamin\":\"Laki-laki\",\"umur\":31,\"membership_program\":12,\"using_reward\":\"No\",\"pembayaran\":\"Credit Card\",\"Subscribe_brochure\":\"Email\",\"harga_per_bulan\":50000,\"jumlah_harga_langganan\":600000}"
```

**Expected response:**
```json
{
  "churn_prediction": 0,
  "churn_label": "No Churn",
  "confidence": 0.8523,
  "probabilities": {
    "0": 0.8523,
    "1": 0.1477
  },
  "model_version": "20260704_230843"
}
```

---

### 5. Open Dashboard (10 seconds)

Jalankan local server untuk dashboard (menghindari issue CORS):

```bash
cd dashboard
python -m http.server 3000
```

Buka browser dan kunjungi: **http://localhost:3000**

**You'll see:**
- ✅ System health status
- ✅ Total predictions count
- ✅ Average confidence score
- ✅ Prediction distribution charts

---

### 6. Run Tests (20 seconds)

```bash
pytest tests/test_api.py -v
```

**Expected:**
```
======================== 13 passed, 7 skipped ========================
```

(7 tests skipped karena butuh running API server)

---

## 🎯 Quick API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get churn prediction |
| `/health` | GET | Check system status |
| `/metrics` | GET | View aggregated metrics (requires API key) |
| `/docs` | GET | Interactive API documentation |

---

## 📝 Example Payloads

### Valid Request (No Churn)
```json
{
  "Jenis_kelamin": "Perempuan",
  "umur": 25,
  "membership_program": 24,
  "using_reward": "Yes",
  "pembayaran": "Bank Transfer",
  "Subscribe_brochure": "WhatsApp",
  "harga_per_bulan": 75000,
  "jumlah_harga_langganan": 1800000
}
```

### Valid Request (Likely Churn)
```json
{
  "Jenis_kelamin": "Laki-laki",
  "umur": 65,
  "membership_program": 3,
  "using_reward": "No",
  "pembayaran": "Credit Card",
  "Subscribe_brochure": "Email",
  "harga_per_bulan": 150000,
  "jumlah_harga_langganan": 450000
}
```

---

## 🔍 Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "20260704_230843",
  "uptime_seconds": 3847.52
}
```

---

## 📊 Metrics (Protected)

```bash
curl http://localhost:8000/metrics \
  -H "X-API-Key: dev-key-change-in-production"
```

**Response:**
```json
{
  "total_requests": 42,
  "predictions_distribution": {
    "0": 28,
    "1": 14
  },
  "average_confidence": 0.8241,
  "last_request_at": "2026-07-04T23:15:42.123456"
}
```

---

## ⚠️ Troubleshooting

### Model Not Found
```
FileNotFoundError: Model file not found
```
**Fix:** Run `python train.py` first

### Port Already in Use
```
ERROR: [Errno 10048] error while attempting to bind on address
```
**Fix:** Change port: `--port 8001` or kill existing process

### Test Failures
```
ModuleNotFoundError: No module named 'src'
```
**Fix:** Run from project root directory

---

## 🎓 What's Next?

- ✅ Read full documentation: `README.md`
- ✅ Understand refactoring: `REFACTORING_SUMMARY.md`
- ✅ Explore test suite: `tests/test_api.py`
- ✅ Customize config: `.env`
- ✅ Deploy to production: See README Docker section

---

## 💡 Pro Tips

1. **Auto-reload:** Use `--reload` flag during development
2. **Production mode:** Remove `--reload`, add `--workers 4`
3. **Monitor logs:** Check `logs/predictions.jsonl`
4. **Test coverage:** Run `pytest --cov=src --cov=app`
5. **API docs:** Always available at `/docs` endpoint

---

**Happy Predicting! 🎯**
