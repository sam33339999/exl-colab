# KV cache 基準：沒命中 vs 有命中

2026-10-08，這台 Colab（A100-SXM4 40GB、83 GiB RAM、swap 0；規格見 README 第 1 節）。當時載入的是 Swift-1.5 EXL3，不是現在的 Coder390。服務用當時 `config.yml` 的推薦值：

* `max_seq_len: 262144`
* `cache_size: 393216`（Q8，主模型與 MTP 草稿都是 Q8）
* `max_batch_size: 8`
* `draft_mode: mtp`
* `memory.sysmem_kv_cache: 24576`
* `memory.sysmem_recurrent_cache: 12288`

兩組都是串流 chat completion，`temperature 0.7`，關閉 thinking，停用 EOS，所以每條都生成滿 **256** token（`finish=length`）。TTFT 是客戶端收到第一個 token 的時間。Prefill / decode tok/s 來自 API timings 的 `prompt_per_second` 與 `predicted_per_second`。

Page 大小是 **256** token。只有寫滿的 page 能留下重用；當前這頁的尾巴一定重算。

---

## 1. 沒有命中 KV

**這整節的前提是 KV 命中 = 0。** 八條前文從第一個 token 就不同，沒有可接的前綴，每條約 2424 token 全部重跑 prefill。Cache 池是開著的，只是對不上。

| | Prefill | Decode | TTFT |
|---|---|---|---|
| 單流 | **2287 tok/s**（2424 token，1.06 s） | **63.8 tok/s**（256 token，4.01 s） | **1.16 s** |
| 8 併發 | 每條平均 **932 tok/s**（640–1769） | 每條平均 **21.0 tok/s**（16.8–26.4） | 平均 **7.49 s**（5.08–9.34 s） |

8 併發牆鐘 **19.9 s**，共 2048 個生成 token。Prefill 幾乎逐條排隊，所以 TTFT 從 5.08 s 排到 9.34 s，不是常態分布。8 條合計 19392 個新 token，以最後一條 TTFT 估算，整批 prefill 約 **2076 tok/s**，接近單流。

MTP 接受率：單流 137/283（48%）。8 併發約 29–42%。

---

## 2. 有命中 KV

同一條前文先跑完一次（暖機，該次命中 0），再測單流與 8 併發。8 條用的是同一段文字。

Prompt **2353** token。命中 **2304 / 2353（97.9%）**，也就是 9 個滿頁。剩下 **49** token 是最後不滿一頁的尾巴，每條仍重算，耗時 **110 ms**。這 49 token 少於引擎報告 prefill 速率的門檻（256），所以不把該次的 tok/s 當成 prefill 速度。

| | 重算 | Decode | TTFT |
|---|---|---|---|
| 單流 | 49 token / 110 ms（前面 2304 命中） | **67.9 tok/s**（256 token，3.77 s） | **0.24 s** |
| 8 併發 | 每條同樣 49 token / 110 ms，命中都是 97.9% | 每條平均 **24.4 tok/s**（22.0–26.4） | 平均 **1.24 s**（0.45–1.94 s） |

8 併發牆鐘 **12.0 s**。Decode 和沒命中時同一量級（單流 64–68 tok/s，8 人每條約 21–24 tok/s）。差在 TTFT：單流 1.16 s → 0.24 s，8 人平均 7.49 s → 1.24 s。

這個混合模型要接回前綴，除了 KV page，還要有對應的 Gated DeltaNet 檢查點。只有 page、沒有那個狀態時，一樣會整段重算。

---

## 3. 對照

| | 沒命中，單流 | 命中 97.9%，單流 | 沒命中，8 併發 | 命中 97.9%，8 併發 |
|---|---|---|---|---|
| TTFT | 1.16 s | 0.24 s | 7.49 s | 1.24 s |
| 要重算的 prompt | 2424 token | 49 token | 每條 2424 | 每條 49 |
| Decode | 63.8 tok/s | 67.9 tok/s | 21.0 tok/s | 24.4 tok/s |
| 牆鐘 | 5.1 s | 3.9 s | 19.9 s | 12.0 s |
