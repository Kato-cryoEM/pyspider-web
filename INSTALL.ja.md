# インストールガイド (日本語)

2 つのインストール経路があります。用途で選んでください。

| 経路 | 対象 | 必要環境 |
|---|---|---|
| **A. Web ビルド (Pyodide)** | 学生・講師のメイン想定。チャレンジ/チュートリアルを触るだけ。 | モダンブラウザのみ |
| **B. ローカルサーバー (FastAPI)** | 開発、独自プラグイン、大容量データ、オプションの CUDA | Python 3.11+、Node 20+ |

以降は macOS と Windows を並行して記載します。英語版は [INSTALL.md](INSTALL.md)。

---

## A. Web ビルド (Pyodide) — 最も簡単

### A-1. ビルド済みリリースから起動

既に `pyspider_web_dist/frontend/dist/` 一式がある前提 (ダウンロード or `npm run build` 済み)。あとは任意の静的 HTTP サーバーで公開するだけ。

#### macOS

```sh
cd パス/to/pyspider_web_dist/frontend/dist
python3 -m http.server 8080
```

Safari / Chrome / Firefox で http://127.0.0.1:8080/ を開く。

最近の macOS には Python 3 が標準装備。なければ https://www.python.org/downloads/mac-osx/ からインストール。

#### Windows

```powershell
cd パス\to\pyspider_web_dist\frontend\dist
python -m http.server 8080
```

Edge / Chrome / Firefox で http://127.0.0.1:8080/ を開く。

`python` がない場合: https://www.python.org/downloads/windows/ からインストール。**"Add python.exe to PATH" にチェック** を入れることを忘れずに。

### A-2. 初回ロード時の挙動

- ページを開くと Pyodide + numpy / scipy / Pillow / scikit-image (合計 ~45 MB) をダウンロード。以降はブラウザキャッシュで高速。
- 起動ログが `ready` まで進むと UI が現れます。
- ブラウザサンドボックス内 (Pyodide 仮想 FS) のみで動作。ディスクに書き込みません。

### A-3. ソースからビルド

リポジトリを git clone したけど `dist/` が無い場合:

```sh
# macOS / Windows 共通。Node 20+ 必須。
cd frontend
npm install
npm run build
cd ..
bash pyodide_poc/build_bundle.sh   # frontend/dist/bundle.zip を生成
```

Windows ではこのシェルスクリプトを Git Bash か WSL で実行する、または手動で:

```powershell
Remove-Item -Force frontend\dist\bundle.zip -ErrorAction SilentlyContinue
powershell -Command "Compress-Archive -Path backend,pyspider -DestinationPath frontend\dist\bundle.zip -Force"
```

---

## B. ローカルサーバー (FastAPI) — 開発モード

### B-1. Python と Node をインストール

#### macOS

```sh
# Homebrew 経由
brew install python@3.12 node@20
```

#### Windows

- Python: https://www.python.org/downloads/windows/ ("Add to PATH" にチェック)
- Node: https://nodejs.org/ (LTS)

新規ターミナルで確認:

```sh
python --version    # 3.11+
node --version      # 20.x+
npm --version
```

### B-2. Python 依存をインストール

```sh
cd パス/to/pyspider_web_dist
python -m venv .venv

# venv 有効化:
#   macOS / Linux
source .venv/bin/activate
#   Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### B-3. バックエンド起動

```sh
uvicorn backend.main:app --reload --port 8000
```

このターミナルは開きっぱなしに。

### B-4. フロントエンド (開発モード) 起動

別ターミナルで:

```sh
cd パス/to/pyspider_web_dist/frontend
npm install        # 初回のみ
npm run dev
```

http://127.0.0.1:5173/ をブラウザで開く。

### B-5. オプション — CUDA 加速

バックエンドの venv で:

```sh
pip install cupy-cuda12x    # CUDA 12
# または
pip install cupy-cuda11x    # CUDA 11
```

UI 右上のドロップダウンで実行時に CPU ↔ CUDA 切替可能。

---

## よくあるトラブル

### `python` が見つからない (Windows)

インストーラで PATH に追加しなかった可能性大。インストーラを再実行して **"Add python.exe to PATH"** にチェックを入れるか、`C:\Users\<ユーザー名>\AppData\Local\Programs\Python\Python312\` を手動で PATH に追加。

### `python3 -m http.server` が `permission denied` (macOS)

1024 未満のポートは管理者権限が必要。8080 / 8000 / 5173 を使いましょう。

### 初回ロードが遅い (ブラウザ)

正常です。Pyodide + パッケージ合計 ~45 MB のダウンロードが初回だけ発生。以降はキャッシュで瞬時。

### "worker not initialised" エラー (Pyodide モード)

ページを強制リロード (⌘-Shift-R / Ctrl-Shift-R)。Worker がロード失敗 — 大体サーバーが `pyodideWorker.js` か `bundle.zip` を返していないケース。両 URL が 200 を返すか確認:
- `http://127.0.0.1:8080/pyodideWorker.js`
- `http://127.0.0.1:8080/bundle.zip`

### `npm install` が Windows で失敗

`frontend\node_modules` と `frontend\package-lock.json` を削除してもう一度 `npm install`。Node 20+ であることも確認 (古い Node は必須 API 欠落)。

### ローカルサーバーモードで CORS エラー

バックエンド 8000 ポート、フロントエンド 5173 ポートで両方起動しているか確認。Vite のプロキシ設定はこの 2 ポート前提です。変更したい場合は `frontend/vite.config.ts` を編集。

---

## ファイル配置

| | macOS | Windows |
|---|---|---|
| アプリ本体 | `/Applications/pyspider_web_dist/` 等任意 | 任意 (長パス回避のため浅めがおすすめ) |
| Python venv | アプリ下 `.venv/` | アプリ下 `.venv\` |
| フロントエンドビルド | `frontend/dist/` | `frontend\dist\` |
| バックエンド静的マウント | `backend/_static/` (FastAPI が `/` で配信) | 同じ |
| ブラウザキャッシュ | ブラウザ標準 | ブラウザ標準 |

---

## アンインストール

### Web ビルド
ディレクトリを削除するだけ。ブラウザキャッシュは数週間で自動クリア、または手動でブラウザ設定から削除。

### ローカルサーバー
```sh
# venv 無効化 & 削除
deactivate
rm -rf .venv            # macOS / Linux
Remove-Item -Recurse -Force .venv   # Windows

# node_modules も消すなら
rm -rf frontend/node_modules
```

アプリディレクトリ外にはファイルを一切書き込みません (ブラウザキャッシュ除く)。システムレジストリ・サービス等への登録もなし。
