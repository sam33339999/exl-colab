# Qwen3.8-27B EXL3 推論伺服器架構與完整實驗紀錄

本專案使用 **ExLlamaV3 1.5.4** 作為推論後端，搭配 **TabbyAPI** 提供相容於 OpenAI 的 REST API 服務，部署於 NVIDIA A100-SXM4-40GB 上，針對自定義量化模型 `sam33339999/Swift-1.5-Qwen3.8-27b-Uncensored-exl3-3.5bpw` 進行多用戶併發優化與投機解碼（Speculative Decoding）實驗。

---

## 1. 系統硬體與執行環境

2026-10-08 在這台 Google Colab 上實測。可用記憶體與硬碟會隨工作階段變動。

| 項目 | 規格 / 版本 | 備註 |
|---|---|---|
| **作業系統** | Ubuntu 24.04.4 LTS (Noble Numbat) | Linux 6.6.122+ x86_64（Google Colab） |
| **GPU** | NVIDIA A100-SXM4-40GB | 40960 MiB，Compute Capability 8.0（Ampere） |
| **CUDA / Driver** | CUDA 13.0 / Driver 580.82.07 | PyTorch `2.11.0+cu130` |
| **CPU** | Intel Xeon @ 2.20GHz | 1 socket、6 cores、12 threads |
| **記憶體** | 83 GiB | 測量時可用約 77 GiB。Swap **0** |
| **硬碟** | overlay 236 GB | 測量時已用 63 GB、可用 174 GB |
| **Python** | Python 3.13.15 | 搭配 `uv` 管理獨立虛擬環境 `.venv` |
| **PyTorch** | 2.11.0+cu130 | 支援 CUDA 13.0 |
| **推論引擎** | ExLlamaV3 1.5.4 (`cu132.torch2.11.0-cp313`) | 預編譯 wheel 整合 Flash Linear Attention |

---

## 2. 推論 Runtime 選型分析

針對多用戶、低延遲與高顯存利用率的推論需求，各 Runtime 的比較結論如下：

| Runtime | 適用場景 | 本機 A100 40GB 實測與評估 |
|---|---|---|
| **ExLlamaV3 + TabbyAPI** | **低 VRAM 佔用、極致單/多用戶速度** | ⭐ **最終選用**。內建 Continuous Dynamic Batching 與 Paged KV Cache，量化精度（EXL3）在 3.5 bpw 下權重僅佔 15GB，預留充沛顯存給 KV Cache 與投機解碼。 |
| **vLLM** | 大規模雲端叢集、超高併發（幾十～上百用戶） | 生態成熟，但在 3~4 bit 非對稱自訂量化格式（如 EXL3）的支援度與極限吞吐不如原生 ExLlamaV3。 |
| **llama.cpp** | GGUF 格式、CPU+GPU 混跑、邊緣設備 | 跨平台相容性佳，但在 A100 高頻寬 GPU 上，EXL3/Flash-Attention 的純算力吞吐與動態批處理略遜於 ExLlamaV3。 |
| **SGLang** | 複雜多輪對話、Prompt Cache 重複率高 | 架構與 vLLM 類似，對 RadixAttention 支援佳。 |

> **迷思釐清**：ExLlama 早期版本僅偏向單人聊天，但 **ExLlamaV3 已經原生支援 Continuous Dynamic Batching**。透過 TabbyAPI 封裝後，能同時平行批次處理多個請求，並非只能單人使用。

---

## 3. 模型規格

- **主模型**：`sam33339999/Swift-1.5-Qwen3.8-27b-Uncensored-exl3-3.5bpw`
  - **基礎架構**：`Qwen3_5ForConditionalGeneration`（Hybrid Linear Attention + Full Attention，每 4 層有 1 層為 Full Attention，其餘為線性注意力，大幅壓低 KV Cache 記憶體增長）。
  - **特性**：Multimodal (Vision/Video) 支援、Abliterated (Uncensored)、原生包含 MTP (Multi-Token Prediction) 模組。
  - **模型大小**：~15 GB（EXL3 3.5bpw）。
- **投機解碼草稿模型 (Draft)**：`z-lab/Qwen3.8-27B-DFlash2`
  - **架構**：`DFlash2DraftModel`（Block-Diffusion 草稿預測，BF16，~3.6 GB）。

---

## 4. 關鍵實驗數據與基準測試 (Benchmarks)

測試腳本：`test_concurrency.py` 與 `bench.sh`（均包含冷啟動暖機流程）。

這一節的數字是在舊設定下記的：`max_seq_len: 32768`、`cache_size: 131072`、`cache_mode: FP16`。目前的推薦值在第 6.4 節。

### 4.1 多人併發基準測試（Baseline - 無 Draft）

| 測試項目 | 產生 Token 數 | 耗時 | 吞吐量 (Aggregate tok/s) | 每人均速 | VRAM 佔用 |
|---|---|---|---|---|---|
| **單請求 (Single, temp 0.7)** | 61 tok | 1.4s | **42.8 tok/s** | 42.8 tok/s | ~22.9 GB |
| **8 人併發 (8 Concurrent, temp 0.7)** | 632 tok | 3.4s | **186.0 tok/s** | ~23.3 tok/s | ~22.9 GB |

> **說明**：8 人同時送出請求時，總吞吐量提升了 **4.34 倍**，證明 Dynamic Batching 在 A100 上達到極高算力利用率。

---

### 4.2 投機解碼方案對比（DFlash2 vs MTP vs Baseline）

| 測試情境 | 關閉 Draft (Baseline) | 模型內建 MTP | DFlash2 (Block Diffusion) | 評析與現象說明 |
|---|---|---|---|---|
| **中文單人 (temp 0.7)** | 44.4 tok/s | **52.0 tok/s** (+17%) | 37.8 tok/s (-15%) | DFlash2 接受率僅 15~25%，誤判回滾抵銷優勢；MTP 穩定。 |
| **中文 8 人併發 (temp 0.7)** | **173.0 tok/s** | 153.2 tok/s | 128.2 tok/s | **高併發下 GPU 算力已飽和**，任何投機驗證開銷都會降低整體 Throughput。 |
| **代碼生成單人 (greedy)** | 48.3 tok/s | 109.0 tok/s (+125%) | **156.2 tok/s** (+223%) | 結構化代碼預測度高，DFlash2 接受率達 75%，飆至 **156.2 T/s**。 |
| **英文回答單人 (greedy)** | 48.4 tok/s | 97.9 tok/s (+102%) | **118.2 tok/s** (+144%) | 邏輯性英文預測接受率達 53%，DFlash2 勝出。 |
| **顯存佔用 (VRAM)** | **22.8 GB** | **28.6 GB** | **38.1 GB** (臨界 40GB) | DFlash2 需額外載入 3.6GB BF16 權重與草稿上下文，長上下文恐 OOM。 |

---

## 5. MTP 推測步數 (Draft Tokens) 最佳化與甜蜜點

在 TabbyAPI / ExLlamaV3 中，可透過 `draft_num_tokens` 設定每次推測產生的 Token 數。

### 為什麼會有甜蜜點？
投機解碼的加速比公式取決於：
1. **接受率衰減**：第 $k$ 個推測 Token 的接受機率隨步數呈指數下降 ($p^k$)。
2. **驗證與草稿前向開銷**：步數過長時，產生草稿與一次性驗證大型注意力矩陣的計算量顯著增加。

### ExLlamaV3 原始碼依據與實測結論：
在 ExLlamaV3 的 `qwen3_5_mtp.py` 原始碼中明確標註：
```python
"default_draft_size": 4,  # best measured performance
```
- **單人程式碼 / 邏輯推理 (Greedy, 低溫)**：甜蜜點為 **3 ~ 4 tokens**。接受率維持在 60% 以上，速度提升最明顯（可達 2~2.5x）。
- **多人併發 / 開放式中文對話 (Sampling, temp > 0.6)**：甜蜜點為 **1 ~ 2 tokens**（或甚至直接使用 `dynamic_draft: true` 讓系統自動根據接受率下調，或完全關閉 Draft）。
- **超過 5 tokens 以上**：邊際效益遞減，多數後續 Token 被 Reject，導致額外耗損 GPU 頻寬與顯存，速度反而下滑。

---

## 6. 專案目錄結構與腳本詳細解析

### 6.1 目錄架構總覽

```
/content/exl3-server/
├── start.sh                 # [Shell]  伺服器主控、自動環境配置、模型下載與守護行程腳本
├── stop.sh                  # [Shell]  優雅停機、釋放 GPU 顯存與清理進程腳本
├── bench.sh                 # [Shell]  全情境性能基準測試套件 (涵蓋中文、併發、代碼、英文)
├── test_concurrency.py      # [Python] 多線程併發壓力測試工具 (計算單人 tok/s 與總吞吐量)
├── config.yml               # [YAML]   每次啟動都會套用的 TabbyAPI 設定（推薦值）
├── docs/kv-cache-benchmark.md  # [Doc]  KV 沒命中 / 有命中的 prefill、decode、TTFT
├── .gitignore               # [Git]    倉庫防護設定 (排除 19GB 權重、虛擬環境、金鑰與日誌)
├── README.md                # [Doc]    繁體中文專案技術文件與實驗記錄
├── server.log               # [Log]    伺服器執行日誌 (包含每次請求之 Token 產速，已被 gitignore)
├── server.pid               # [Runtime]背景進程 PID 記錄檔 (已被 gitignore)
├── models/                  # [Weights]模型權重存放目錄 (已被 gitignore)
│   ├── Swift-1.5-Qwen3.8-27b-exl3-3.5bpw/ # 主模型 (~15GB)
│   └── Qwen3.8-27B-DFlash2/               # DFlash2 草稿模型 (~3.6GB)
└── tabbyAPI/                # [Submodule/Upstream] 推論引擎服務代碼與環境 (已被 gitignore)
    ├── api_tokens.yml       # 自動生成的 API/Admin 金鑰 (已被 gitignore)
    └── .venv/               # Python 3.13 + Torch 2.11 + cu130 虛擬環境
```

---

### 6.2 Shell 腳本詳細說明

#### 1. [`start.sh`](file:///content/exl3-server/start.sh) — 伺服器核心啟動與守護腳本
本專案的主要進入點，具備全自動化環境配置、動態模式切換與背景守護能力。

* **核心功能流程**：
  1. **自動環境建置 (`ensure_setup`)**：檢查系統是否安裝 `uv`，若無則自動下載安裝；自動檢測 `tabbyAPI` 目錄（缺失時自動從 GitHub clone）；若無 `.venv` 則自動建立 Python 3.13 虛擬環境並安裝 ExLlamaV3 (`.[cu13]`)。
  2. **模型自動補全**：檢測 `models/` 目錄下是否存在 `.safetensors` 權重檔，缺失時自動透過 Hugging Face Hub 下載指定模型。
  3. **套用自訂設定**：每次啟動都把倉庫的 `config.yml` 複製成 `tabbyAPI/config.yml`。要改上下文、cache 或記憶體層，改倉庫這份再重開。
  4. **投機解碼模式即時切換**：讀取環境變數 `DRAFT`，在複製後改寫 `draft_mode`（`dflash2`、`mtp`、`off`）。未設定時沿用 `config.yml` 裡的 `mtp`。
  5. **背景守護與 PID 管理 (`--bg`)**：使用 `setsid nohup` 脫鉤終端並在背景啟動服務，自動將 PID 寫入 `server.pid`，所有輸出重新導向至 `server.log`。已在跑時會拒絕再起一個。
  6. **健康檢查輪詢 (Healthcheck Loop)**：背景啟動後自動輪詢 `http://127.0.0.1:5000/health`（最長等待 180 秒），待模型加載完畢並呈現 Ready 狀態時，自動從 `api_tokens.yml` 抓取並顯示 API Key 與 Admin Key。
* **支援參數與環境變數**：
  | 參數 / 變數 | 說明 | 範例 |
  |---|---|---|
  | *(無參數)* | 前景直接執行，日誌直接輸出到終端（適合開發除錯） | `./start.sh` |
  | `--bg` | 一鍵背景啟動，並自動監聽 Ready 狀態 | `./start.sh --bg` |
  | `--stop` | 調用 `stop.sh` 停止正在背景運行的服務 | `./start.sh --stop` |
  | `--download` | 僅執行模型下載程序，不啟動伺服器 | `./start.sh --download` |
  | `--help`, `-h` | 印出用法 | `./start.sh --help` |
  | `DRAFT=<mode>` | 指定投機解碼模式：`dflash2`（草稿模型）、`mtp`（內建多Token預測）、`off`（關閉） | `DRAFT=mtp ./start.sh --bg` |

---

#### 2. [`stop.sh`](file:///content/exl3-server/stop.sh) — 優雅停機與顯存安全釋放腳本
專門解決推論服務終止時最常見的「CUDA 顯存未釋放」與「殘留 Zombie 進程」問題。

* **核心功能流程**：
  1. **雙重 PID 識別**：優先讀取 `server.pid` 記錄的 PID；若 PID 檔案遺失或損壞，自動透過 `pgrep -f "tabbyAPI.*main.py"` 掃描正在執行的推論進程。
  2. **優雅終止 (Graceful Shutdown)**：向目標進程發送 `SIGTERM` 信號，觸發 TabbyAPI 內部的模型卸載程序（Unload Model），確保 CUDA Context 乾淨清除並釋放約 38GB 的 GPU 顯存。
  3. **安全等待循環**：每秒偵測進程存活狀態，最多等待 15 秒以允許模型釋放完成。
  4. **超時強制終止 (Force Kill)**：若超過 15 秒進程仍未退出，自動升級為 `SIGKILL` (`kill -9`) 強制回收資源。
  5. **環境清理**：清理 `server.pid` 檔案並輸出完成狀態。
* **使用方式**：
  ```bash
  ./stop.sh
  ```

---

#### 3. [`bench.sh`](file:///content/exl3-server/bench.sh) — 全自動化綜合性能測試套件
一鍵化測試伺服器在各類典型任務情境下的表現，並自動解析日誌輸出標準評測數據。

* **測試涵蓋情境**：
  1. **GPU 與注意力暖機 (Warmup)**：自動發送 2 筆輕量請求進行預熱，排除冷啟動（Cold Start）對測試數據的干擾。
  2. **中文開放對話測試**：呼叫 `test_concurrency.py 8`，以採樣模式 (`temp: 0.7`) 測試 8 人同時提問台灣地理資訊的彙總吞吐量。
  3. **程式碼生成測試 (Greedy)**：請求 Python 快速排序（Quicksort）實作，限制 `max_tokens: 300`, `temperature: 0`，驗證邏輯與符號結構化輸出的極限單人加速比。
  4. **英文長文技術問答 (Greedy)**：請求英文解釋 TCP Handshake 原理，限制 `max_tokens: 300`, `temperature: 0`。
  5. **顯存實時量測**：透過 `nvidia-smi` 查詢當前顯存使用數值，記錄 VRAM 開銷。
* **使用方式**：
  ```bash
  ./bench.sh
  ```

---

### 6.3 Python 腳本詳細說明

#### 1. [`test_concurrency.py`](file:///content/exl3-server/test_concurrency.py) — 多線程併發壓力測試工具
輕量化、零第三方套件依賴的原生 Python 併發測試工具。

* **設計特色與核心原理**：
  1. **零外部依賴**：完全採用 Python 內建標準庫（`urllib.request`、`concurrent.futures`、`json`、`time`、`re`、`pathlib`），無需額外安裝 `requests` 或 `aiohttp`，任何環境皆可開箱即用。
  2. **自動金鑰認證**：自動從 `tabbyAPI/api_tokens.yml` 透過正規表達式提取 `api_key`，無需使用者手動複製貼上。
  3. **防止 Prompt 快取作弊**：內建台灣 12 個主要縣市列表（台北、台中、高雄、台南、新竹、花蓮等），每個併發請求動態替換提問縣市，確保模型真實進行 Prefill 與 Decode 計算，而非單純命中快取。
  4. **雙階段對比測試**：
     * **Phase 1 (Single Request)**：執行單條請求，取得基線單人速度與延遲。
     * **Phase 2 (N Concurrent Requests)**：利用 `ThreadPoolExecutor` 同時派發 $N$ 條請求，模擬多用戶真實競爭情況。
  5. **精確度量指標**：
     * 單一請求：記錄產生之 Token 數、耗時（秒）、單人生成速率（`tok/s`）及前 50 字內容預覽。
     * 併發整體：統計全體產生之總 Token 數、總運行時間，並精確計算**彙總吞吐量 (Aggregate tok/s)**。
* **使用範例**：
  ```bash
  # 預設執行 6 人併發
  python3 test_concurrency.py

  # 指定 8 人併發測試
  python3 test_concurrency.py 8

  # 測試極限 16 人併發
  python3 test_concurrency.py 16
  ```

---

### 6.4 設定檔與環境防護說明

#### 1. [`config.yml`](file:///content/exl3-server/config.yml) — 本機推薦值

依第 1 節這台 Colab 主機寫入的日常設定。模型原生上下文是 **262,144**，這份設定不開 YaRN。

* `max_seq_len: 262144`：單次請求上限，等於模型 `config.json` 的 `max_position_embeddings`。
* `cache_size: 393216`：共用 Paged KV cache，等於 1.5 份 262K。一條請求可以吃滿 262K，其餘約 13 萬 token 給同時在跑的較短任務。短任務只占用實際頁數（256 token 一頁）。以 Q8 估算，載入後 GPU 大約還剩 7.3 GB。
* `cache_mode: Q8`、`draft_cache_mode: Q8`：主模型 16 層 full attention 與 MTP 那 1 層的 KV 都用 8-bit。
* `max_batch_size: 8`：同時生成的上限。
* `memory.sysmem_kv_cache: 24576`：24 GB pinned RAM，承接從 GPU 擠出的 KV page。命中時拷回 GPU，不再重跑 prefill。這不會把單條上下文拉過 `cache_size`。硬碟沒有 KV 層。
* `memory.sysmem_recurrent_cache: 12288`：12 GB，大約 80 個 Gated DeltaNet 檢查點（每個約 148 MB）。這個混合模型要接回前綴，KV 和 recurrent 狀態要一起留。
* `memory.sysmem_multimodal_cache: 1024`：重複出現的圖片不必重跑 vision tower。
* `draft_mode: mtp`：使用模型內建 MTP，並開啟 `dynamic_draft: true`。

`./start.sh` 每次啟動都會把這份 `config.yml` 複製到 `tabbyAPI/config.yml` 再啟動。執行中的服務要 `./start.sh --stop` 之後再啟動才會吃到新值。單條要超過 262K 時另開一份設定：YaRN `factor` 4、Q4 cache、`max_batch_size: 1`。靜態 YaRN 會讓短文變差，而且 Q4 的 1M cache 在這張 40GB 上只剩約 1.7 GB。

KV cache **沒有時間過期**。請求結束後，寫滿的 page（256 token）一直留著，直到池子不夠才淘汰：先拿空白頁和已經斷掉的頁，再從最久沒被用到的舊對話尾巴開始砍。從 GPU 擠出的完整 page 會進 24 GB 的 RAM 第二層，之後命中就拷回，不必重跑 prefill。RAM 那層也是滿了才丟。重開伺服器才會一次清空。

這個混合模型要接回前綴，還得留著對應的 Gated DeltaNet 檢查點（`sysmem_recurrent_cache`，12 GB，滿了才丟）。KV page 還在、檢查點沒了，那段前綴一樣整段重算。數字見 [`docs/kv-cache-benchmark.md`](docs/kv-cache-benchmark.md)：沒命中時單流 TTFT 1.16 s、8 人平均 7.49 s；完整頁命中（97.9%，尾巴 49 token 仍重算）時單流 TTFT 0.24 s、8 人平均 1.24 s。Decode 兩邊都在同一量級。

#### 2. [`.gitignore`](file:///content/exl3-server/.gitignore) — 倉庫輕量化與機密安全防護
確保代碼倉庫體積保持在 36KB 的關鍵防線：
* **排除巨型模型權重**：排除 `models/`、`*.safetensors`、`*.bin` 等（避免誤將 19GB 權重推進 Git）。
* **排除子專案與虛擬環境**：排除 `tabbyAPI/` 上游代碼庫與其內部的 `.venv/`。
* **排除機密金鑰**：排除 `tabbyAPI/api_tokens.yml`，避免 API Key 與 Admin Key 洩漏。
* **排除運行日誌與二進位**：排除 `server.log`、`server.pid` 及下載的 `*.deb` 安裝包。

---

## 7. 快速操作指南

### 7.1 啟動與切換投機解碼模式

一鍵背景啟動會套用倉庫的 `config.yml`（Q8、262K、cache 393216）：

```bash
cd /content/exl-colab

./start.sh --help
./start.sh --bg

# 1. 以 DFlash2 模式啟動 (適合代碼、英文、單人極致速度)
DRAFT=dflash2 ./start.sh --bg

# 2. 以 MTP 模式啟動 (最均衡，省顯存且通用性佳)
DRAFT=mtp ./start.sh --bg

# 3. 關閉 Draft 模式 (最適合多人高併發與長上下文對話)
DRAFT=off ./start.sh --bg

# 停止伺服器 (兩種方式皆可)
./stop.sh
# 或
./start.sh --stop
```

### 7.2 執行壓力與併發測試

```bash
# 測試 8 人併發
python3 test_concurrency.py 8

# 執行全情境綜合性能測試
./bench.sh
```

### 7.3 API 存取方式 (OpenAI 相容)

- **Endpoint**: `http://127.0.0.1:5000/v1/chat/completions`
- **金鑰**: 存放於 `tabbyAPI/api_tokens.yml`（啟動時會自動輸出）。
- **呼叫範例**：
  ```bash
  KEY=$(grep api_key tabbyAPI/api_tokens.yml | awk '{print $2}')
  curl -s http://127.0.0.1:5000/v1/chat/completions \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Swift-1.5-Qwen3.8-27b",
      "messages": [{"role": "user", "content": "你好！"}],
      "temperature": 0.7
    }'
  ```

---

## 8. 多用戶外部連線建議 (Google Colab 環境)

由於當前環境為 Google Colab 虛擬機，缺乏公網 IP。若要分享給團隊多用戶使用：
1. **Cloudflare Tunnel (推薦)**：免開 port、免固定 IP、自帶 SSL 加密。
   ```bash
   # 下載 cloudflared 並映射 5000 port
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   dpkg -i cloudflared.deb
   cloudflared tunnel --url http://127.0.0.1:5000
   ```
2. **ngrok / localtunnel**：作為替代穿透工具。
