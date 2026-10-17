# RTX 3090 上的 exllamav3_ext 編譯紀錄

2026-10-18 00:01（+0800）這台機器上最後一次成功的 JIT 編譯。服務 MiniCPM-V 4.7 3.5bpw 的 TabbyAPI 用的就是這份 `.so`。

權重目錄的啟動說明在 `/home/sam/MiniCPM-V-4.7-35B-A3B-abliterated-EXL3-3.5bpw/RUN.md`。那份檔不在 git 裡。這份文件是同一套作法的版本庫副本。

## 產物

| 項目 | 值 |
|---|---|
| 檔案 | `~/.cache/torch_extensions/py312_cu130/exllamav3_ext/exllamav3_ext.so` |
| 大小 | 21675376 bytes |
| 時間 | 2026-10-18 00:01:11 +0800 |
| Python | CPython 3.12.13（uv） |
| torch | 2.14.1+cu130 |
| 來源 | `/home/sam/exllamav3-origin`，分支 `feat/minicpm-v4.7`，套件版本 1.6.2 |
| GPU | NVIDIA GeForce RTX 3090，compute capability 8.6 |
| 驅動 | 595.91.07。`nvidia-smi` 回報的 CUDA 是驅動能力，不是拿來編譯的 nvcc |

同一天較早、由 `/home/sam/exllamav3-origin/.venv` 編成的 `.so` 已被這次覆蓋。2026-10-16 那份同大小的備份還在：

`~/.cache/torch_extensions/py312_cu130/exllamav3_ext.so.cuda133-20261016`

## 這次寫進 build.ninja 的命令

`nvcc` 是 `/usr/local/cuda-13.3/bin/nvcc`，版本 V13.3.73（`cuda_13.3.r13.3/compiler.38244171_0`，built Tue Jun 9 2026）。

裝置碼只有一行：

```text
-gencode=arch=compute_86,code=sm_86
```

其餘和上次成功編譯一致的旗標：

```text
-O3 --use_fast_math --compress-mode=size -std=c++20
-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=20012
```

C++ 端是 `c++`，`-fPIC -std=c++20 -Ofast`。連結的是 CUDA 13.3 的 `libcudart`，以及這次 import 那個 venv 裡的 torch：

```text
/home/sam/exl-colab/tabbyAPI/.venv/lib/python3.12/site-packages/torch
```

`--compress-mode=size` 只有 CUDA 13.3 這支 nvcc 的 `--help` 有這個選項時才會被加進去。系統的 `/usr/bin/nvcc` 是 CUDA 12.4（V12.4.131），沒有這個選項。

## 每次 import 都要先設的環境

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
```

這四個變數沒有寫進 shell 啟動檔。`CUDA_HOME` 不設的時候，`/usr/local/cuda` 雖然經 alternatives 指到 13.3，找編譯器的順序仍會落到 `PATH` 上的 `/usr/bin/nvcc`。

| 變數 | 為什麼是這個值 |
|---|---|
| `CUDA_HOME` 和 `PATH` | `find_nvcc` 先看 `PYTORCH_NVCC`，再看 `CUDA_HOME`/`CUDA_PATH` 的 `bin`，最後才是 `PATH`。不設的話 ninja 會把 `nvcc` 改成 `/usr/bin/nvcc`，拿掉 `--compress-mode=size`，並覆蓋上面的 `.so` |
| `TORCH_CUDA_ARCH_LIST=8.6` | 變數沒設時，`exllamav3/util/arch_list.py` 會把最後一個架構加成 `8.6+PTX`。那和這份只有 `sm_86` 的 gencode 不同，ninja 會整包重編。不要寫成 `8.6+PTX` |
| `MAX_JOBS=4` | 這台機器用來編譯的並行數。不要讓 ninja 把記憶體吃滿 |

不要設定 `TORCH_EXTENSIONS_DIR`。快取目錄由 Python 版本和 torch 的 CUDA 標籤決定，這裡是 `py312_cu130`。

## 為什麼每個新行程都會跑 ninja

torch 的 extension 版本記在行程裡。新的 Python 行程看不到上一份版本，所以每次 `import exllamav3` 都會重寫 `build.ninja` 再叫 ninja。命令和上一份完全相同時，ninja 不會重編。有一個旗標、路徑或 nvcc 不一樣，就會重編並換掉 `.so`。

命令裡含有該 venv 的 torch include 和 lib 絕對路徑。下面兩個環境都是 CPython 3.12，所以共用同一個 `py312_cu130` 目錄：

- `/home/sam/exllamav3-origin/.venv`
- `/home/sam/exl-colab/tabbyAPI/.venv`

torch 版本都是 2.14.1+cu130，但前綴不同。2026-10-18 這次重編就是 Tabby 的 venv 第一次 import：ninja 看到 include/lib 路徑從 origin 的 venv 換成 Tabby 的 venv。現在檔案裡記的是 Tabby 那組路徑。下一次從 origin 的 `.venv` import，路徑又對不上，會再整包重編。重編時仍然要帶上面的四個變數，否則會編成 CUDA 12.4。

## 確認手上的 .so 是這一份

```bash
grep -E '^nvcc =|gencode|compress-mode' \
  ~/.cache/torch_extensions/py312_cu130/exllamav3_ext/build.ninja
stat -c '%y %s' ~/.cache/torch_extensions/py312_cu130/exllamav3_ext/exllamav3_ext.so
```

應看到 `nvcc = /usr/local/cuda-13.3/bin/nvcc`、`code=sm_86`、`--compress-mode=size`，以及 21675376 bytes。

Tabby 啟動腳本 `start-minicpm-3090.sh` 在叫 `start.sh` 之前會設定這四個變數。直接跑 `./start.sh` 不會設定它們。

## 不要走的安裝

`start.sh` 在 venv 還沒有 torch 和 exllamav3 時會執行 `uv pip install -e ".[cu13]"`。那個 extra 裝的是 torch 2.11.0+cu130 和上游 wheel exllamav3 1.6.1，而且 venv 會是 Python 3.13。上游 wheel 沒有 `MiniCPMV4_7ForConditionalGeneration`，Python 3.13 也會另外編一份 `py313_cu130`。

這台 3090 的 Tabby venv 是手動裝的，順序如下。torch 要先釘死，後面的 editable 安裝才不會把它換成別的版本。

```bash
cd /home/sam/exl-colab
git clone --branch feat/minicpm-v4.7 https://github.com/sam33339999/tabbyAPI.git tabbyAPI
cd tabbyAPI
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.14.1+cu130'
uv pip install --python .venv/bin/python --no-deps \
  -e /home/sam/exllamav3-origin
uv pip install --python .venv/bin/python \
  'tokenizers>=0.21.1' 'numpy>=2.1.0' rich typing_extensions \
  'safetensors>=0.3.2' ninja pillow pyyaml marisa_trie \
  'pydantic>=2.11,<3' 'llguidance>=1.7.0'
uv pip install --python .venv/bin/python -e /home/sam/exl-colab/tabbyAPI
uv pip install --python .venv/bin/python \
  -e /home/sam/exl-colab/minicpm47-native-stub
```

裝完再 import 一次，讓 ninja 用 13.3 編或核對這份 `.so`：

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
/home/sam/exl-colab/tabbyAPI/.venv/bin/python -c 'import exllamav3, exllamav3.version, torch
print(exllamav3.__file__)
print(exllamav3.version.__version__, torch.__version__, torch.cuda.get_device_name(0))'
```

`exllamav3.__version__` 不存在。版本在 `exllamav3.version.__version__`，應為 `1.6.2`。`__file__` 應為 `/home/sam/exllamav3-origin/exllamav3/__init__.py`。editable 安裝的 `direct_url.json` 記的是 `file:///home/sam/exllamav3-origin`；`importlib.metadata` 的 `locate_file()` 不會指到那個目錄。
