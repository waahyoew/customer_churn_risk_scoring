# Churn Prediction Pipeline — Production-Ready MLOps System

> **Transformasi dari notebook eksperimen menjadi production-grade ML system dengan REST API real-time, monitoring dashboard, dan automated testing.**

---

## 🏗️ Struktur Project

```
churn_prediction_pipeline/
├── app/
│   ├── __init__.py
│   └── main.py                 ← FastAPI: endpoints /predict, /health, /metrics
├── dashboard/
│   └── index.html              ← Web monitoring dashboard real-time
├── data/
│   └── cth_churn_analysis_train.xlsx  ← Dataset (letakkan di sini)
├── logs/
│   └── predictions.jsonl       ← (auto-created) Prediction logs
├── models/
│   ├── final_churn_model.joblib       ← (created by train.py)
│   └── final_churn_model_metadata.json
├── src/
│   ├── __init__.py
│   ├── config.py               ← Centralized config dengan Pydantic Settings
│   ├── data_loader.py          ← Data ingestion & initial preprocessing
│   ├── evaluator.py            ← Model evaluation & artifact saving (+ SHA-256)
│   ├── logger.py               ← Logging setup
│   ├── monitoring.py           ← MLOps: logging, metrics, drift detection
│   ├── pipeline.py             ← Training pipeline orchestration
│   ├── predictor.py            ← Model loader & inference engine
│   ├── preprocessing.py        ← Feature engineering & transformations
│   ├── schemas.py              ← Pydantic input/output validation
│   └── trainer.py              ← Hyperparameter tuning dengan GridSearchCV
├── tests/
│   ├── __init__.py
│   └── test_api.py             ← Comprehensive API tests (pytest)
├── .env                        ← Environment variables configuration
├── .gitignore                  ← Git ignore patterns
├── main.py                     ← Legacy training entry point
├── train.py                    ← Standalone training script (recommended)
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## ✨ Fitur Production-Ready

### 🎯 MLOps Features
- ✅ **SHA-256 Model Integrity Verification** — Checksum validation saat load model
- ✅ **Data Drift Detection** — Baseline dari training class distribution
- ✅ **Prediction Logging** — JSONL format dengan buffered writes
- ✅ **Metrics Aggregation** — Real-time statistics via `/metrics` endpoint
- ✅ **Health Monitoring** — Kubernetes-ready `/health` endpoint
- ✅ **Model Versioning** — Timestamp-based version tracking
- ✅ **Confidence Scoring** — Probability breakdown untuk setiap prediksi

### 🔐 Security & Reliability
- ✅ **API Key Protection** untuk metrics endpoint
- ✅ **CORS Configuration** (restrictive, tidak wildcard)
- ✅ **Pydantic Validation** — Type-safe input/output
- ✅ **Graceful Error Handling** — Proper HTTP status codes
- ✅ **Async-Ready** — FastAPI dengan async/await support

### 🧪 Testing & Quality
- ✅ **Automated Test Suite** — pytest dengan 15+ test cases
- ✅ **Happy Path Tests** — Valid inputs
- ✅ **Edge Case Tests** — Boundary values
- ✅ **Validation Tests** — Error scenarios (422, 403, 503)
- ✅ **Model-Conditional Tests** — Skip jika model belum ada

### 📊 Monitoring & Observability
- ✅ **Real-time Dashboard** — HTML dashboard dengan auto-refresh
- ✅ **Visual Metrics** — Prediction distribution charts
- ✅ **Drift Alerts** — Visual indicators untuk data drift
- ✅ **System Status** — Health, uptime, model version

---

## 🚀 Quick Start

### Tahap 1: Setup Pertama Kali (First Time Setup)
Lakukan langkah ini **hanya sekali** saat baru pertama kali mengunduh project:

1. **Buka Terminal** dan arahkan ke folder project:
   ```bash
   cd e:\ML_AI_engineer\datascientist\Project\churn_prediction_pipeline
   ```
2. **Install Dependencies (Library Python):**
   ```bash
   pip install -r requirements.txt
   ```
3. **Pastikan Dataset Tersedia:**
   Pastikan file `cth_churn_analysis_train.xlsx` ada di dalam folder `data/`.
4. **Training Model (Membuat File Model):**
   ```bash
   python train.py
   ```
   *(Tunggu hingga selesai. File model akan otomatis terbuat di folder `models/`)*

---

### Tahap 2: Menjalankan Sehari-hari (Daily Run)
Setiap kali Anda ingin menggunakan aplikasi, cukup jalankan 2 langkah ini (butuh 2 terminal terpisah):

1. **Jalankan API Server (Terminal 1)**
   Buka terminal di folder `churn_prediction_pipeline`, lalu ketik:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *(Biarkan terminal ini tetap menyala)*

2. **Jalankan Dashboard Web (Terminal 2)**
   Buka terminal baru, masuk ke folder dashboard, dan nyalakan web server:
   ```bash
   cd e:\ML_AI_engineer\datascientist\Project\churn_prediction_pipeline\dashboard
   python -m http.server 3000
   ```
   Lalu buka browser dan kunjungi: **http://localhost:3000**

---

### Contoh Input Simulasi (Untuk Testing Dashboard)
Gunakan kombinasi nilai berikut di dashboard untuk melihat perbedaan prediksi potensi nasabah churn (berhenti berlangganan):

- **🟢 Risiko Churn Rendah (Aman - "No Churn"):**
  - Umur: 35 | Membership: 24 bulan | Reward: Yes | Pembayaran: Credit Card | Harga per bulan: 50000

- **🔴 Risiko Churn Tinggi (Bahaya - "Churn"):**
  - Umur: 22 | Membership: 1 bulan | Reward: No | Pembayaran: Direct Debit | Harga per bulan: 150000

---

## 📡 API Endpoints

| Method | Endpoint | Deskripsi | Auth Required |
|--------|----------|-----------|---------------|
| `POST` | `/predict` | Prediksi churn untuk single customer | No |
| `GET`  | `/health` | Status API dan model | No |
| `GET`  | `/metrics` | MLOps metrics (aggregated stats) | Yes (API Key) |
| `GET`  | `/docs` | Swagger UI (interactive testing) | No |

### `/predict` — Churn Prediction

**Request Body:**
```json
{
  "Jenis_kelamin": "Laki-laki",
  "umur": 31,
  "membership_program": 12,
  "using_reward": "No",
  "pembayaran": "Credit Card",
  "Subscribe_brochure": "Email",
  "harga_per_bulan": 50000,
  "jumlah_harga_langganan": 600000
}
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_label": "No Churn",
  "confidence": 0.8523,
  "probabilities": {
    "0": 0.8523,
    "1": 0.1477
  },
  "model_version": "20260704_143022"
}
```

### `/health` — System Health Check

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "20260704_143022",
  "uptime_seconds": 3847.52
}
```

### `/metrics` — MLOps Metrics

**Headers Required:**
```
X-API-Key: dev-key-change-in-production
```

**Response:**
```json
{
  "total_requests": 1547,
  "predictions_distribution": {
    "0": 1204,
    "1": 343
  },
  "average_confidence": 0.8241,
  "last_request_at": "2026-07-04T14:35:22.123456"
}
```

---

## ⚙️ Configuration

Konfigurasi system menggunakan environment variables via file `.env`:

```env
# API Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
METRICS_API_KEY=dev-key-change-in-production

# Data Configuration
CHURN_DATA_URL=https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx
CHURN_DATA_DIR=data
CHURN_LOCAL_DATA_PATH=data/cth_churn_analysis_train.xlsx

# Training Configuration
CHURN_RANDOM_STATE=57
CHURN_TEST_SIZE=0.2
CHURN_CV_SPLITS=5
CHURN_SCORING_METRIC=f1
```

**Environment Variables Prefix:** Semua variable harus diawali `CHURN_` (kecuali CORS_ORIGINS dan METRICS_API_KEY).

---

## 🧪 Testing Strategy

### Test Coverage

```bash
pytest tests/test_api.py -v --cov=src --cov=app
```

### Test Categories

1. **Health Endpoint Tests** (3 tests)
   - Always available regardless of model state
   - Validates response structure
   - Checks uptime calculation

2. **Metrics Endpoint Tests** (5 tests)
   - API key authentication
   - Response structure validation
   - Non-negative values

3. **Validation Tests** (5 tests)
   - Missing fields → 422
   - Invalid enum values → 422
   - Out-of-range values → 422

4. **Inference Tests** (7 tests, model-dependent)
   - Happy path scenarios
   - Edge cases (zero values, boundaries)
   - Probability sum validation
   - Confidence range checks

### Skipping Model-Dependent Tests

Jika model belum di-train, test yang butuh model akan di-skip otomatis:

```
tests/test_api.py::TestPredictInference::test_predict_valid_input_returns_200 SKIPPED
Reason: Model not loaded — run `python train.py` first.
```

---

## 🛠️ Development Workflow

### 1. Data Exploration (Optional)

Gunakan Jupyter Notebook untuk eksplorasi awal:
- `1011_Production_DS_Style.ipynb`

### 2. Feature Engineering

Edit file `src/preprocessing.py` untuk modifikasi feature engineering logic.

### 3. Model Tuning

Edit hyperparameter grid di `src/config.py`:

```python
DECISION_TREE_PARAM_GRID = {
    'model__max_depth': [4, 8, 16, 32, 64, None],
    'model__min_samples_split': [2, 4, 8, 16, 32],
}
```

### 4. Training

```bash
python train.py
```

### 5. Testing

```bash
# Run all tests
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_api.py::TestPredictInference -v

# Run with coverage
pytest tests/test_api.py --cov=src --cov=app --cov-report=html
```

### 6. API Development

```bash
# Development mode (auto-reload)
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📊 Model Information

- **Algorithm:** Decision Tree / Random Forest (auto-selected via GridSearchCV)
- **Scoring Metric:** F1 Score (balanced for imbalanced churn data)
- **Cross-Validation:** Stratified K-Fold (k=5)
- **Features:** 8 input features (numeric + categorical)
- **Target:** Binary classification (0 = No Churn, 1 = Churn)
- **Split:** 80% train / 20% test, stratified, random_state=57

### Feature List

| Feature | Type | Description |
|---------|------|-------------|
| `Jenis_kelamin` | Categorical | Gender (Laki-laki/Perempuan) |
| `umur` | Numeric | Age (0-120) |
| `membership_program` | Numeric | Months in program (≥0) |
| `using_reward` | Categorical | Uses reward program (Yes/No) |
| `pembayaran` | Categorical | Payment method |
| `Subscribe_brochure` | Categorical | Brochure channel |
| `harga_per_bulan` | Numeric | Monthly price (>0) |
| `jumlah_harga_langganan` | Numeric | Total charges (≥0) |

---

## 🐳 Docker Deployment (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

---

## 🔍 Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model file not found at: models/final_churn_model.joblib.
Please run `python train.py` first.
```

**Solution:** Run training pipeline:
```bash
python train.py
```

### API Connection Error

```
Cannot connect to API. Make sure the server is running at http://localhost:8000
```

**Solution:** Start the API server:
```bash
uvicorn app.main:app --reload
```

### Metrics 403 Forbidden

```
Forbidden: invalid or missing X-API-Key header.
```

**Solution:** Include API key in request:
```bash
curl -H "X-API-Key: dev-key-change-in-production" http://localhost:8000/metrics
```

### Test Failures

```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Ensure you're running tests from project root:
```bash
cd e:\ML_AI_engineer\datascientist\Project\churn_prediction_pipeline
pytest tests/test_api.py -v
```

---

## 📈 Monitoring & Drift Detection

### How Drift Detection Works

1. **Training Baseline:** Class distribution disimpan saat training di `model_metadata.json`
2. **Runtime Monitoring:** Setiap 20 prediksi, sistem compare current distribution vs baseline
3. **Alert Threshold:** Warning jika deviation >25% dari baseline
4. **Visual Indicator:** Dashboard menampilkan alert banner jika drift terdeteksi

### Interpreting Drift Alerts

**Log Warning:**
```
[DRIFT ALERT] Class '0': current=35.00%, baseline=60.00%, deviation=25.00% (threshold=25.00%)
```

**Action Required:**
1. Investigate data quality issues
2. Check if business context has changed
3. Consider model retraining dengan data terbaru

---

## 🎓 Architecture Highlights

### Train-Serve Parity

- **Single Source of Truth:** `src/config.py` digunakan oleh training DAN serving
- **No Duplication:** Feature encoding logic hanya ada di preprocessing pipeline
- **Serialized Pipeline:** Model disimpan sebagai complete pipeline (preprocessing + model)

### Scalability Considerations

- **Stateless Design:** API tidak simpan state, horizontal scaling friendly
- **Async I/O:** FastAPI async untuk concurrent requests
- **Buffered Logging:** Prevents I/O bottleneck (flush setiap 20 predictions)
- **Graceful Degradation:** API tetap up meski model corrupt (503 response)

### Security Best Practices

- **API Key Auth:** Metrics endpoint protected
- **CORS Restrictive:** Tidak wildcard (`*`)
- **Input Validation:** Pydantic dengan strict constraints
- **No Secret Exposure:** Error messages tidak expose internal state
- **Checksum Verification:** SHA-256 untuk detect model tampering

---

## 📝 Migration dari Old Version

Jika migrasi dari struktur lama (`main.py` sebagai training script):

### Yang Berubah:

1. **Config:** `src/config.py` sekarang menggunakan Pydantic Settings
2. **Training:** Gunakan `train.py` (bukan `main.py`)
3. **API:** Sekarang ada folder `app/` dengan FastAPI
4. **Tests:** Sekarang ada folder `tests/` dengan pytest
5. **Environment:** Sekarang ada `.env` untuk configuration

### Backward Compatibility:

- ✅ `main.py` masih berfungsi untuk training (legacy support)
- ✅ `1011_Production_DS_Style.ipynb` tetap kompatibel
- ✅ Model format sama (`.joblib` dengan metadata `.json`)

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/nama-fitur`
2. Write tests first (TDD approach)
3. Implement feature
4. Run tests: `pytest tests/ -v`
5. Update documentation
6. Submit pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🔗 Related Projects

- [Credit Risk Scoring](../credit_risk/) — Referensi architecture pattern yang sama

---

## 📞 Support

Untuk pertanyaan atau issues:
1. Check troubleshooting section di atas
2. Review API docs: `http://localhost:8000/docs`
3. Check logs: `logs/predictions.jsonl` dan `logs/pipeline.log`

---

**Built with ❤️ using FastAPI, scikit-learn, and modern MLOps practices**
