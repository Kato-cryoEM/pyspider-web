# pyspider · ノードベース cryo-EM 学習環境

SPIDER のオペレータ群をベースにした、ブラウザで動く cryo-EM 画像処理学習環境です。React Flow のキャンバス上でノードを組み合わせ、パラメータを調整して Run すると、本物の numpy / scipy のコードが動いて中間結果のプレビューが全ノードに表示されます。

同じコードベースから 2 つの配布形態が生成できます:

- **Web ビルド (Pyodide)** — 静的ファイル (`frontend/dist`) 一式。任意の HTTP サーバーで公開すれば、Python バックエンドが**ブラウザ内で丸ごと動作** (Pyodide Web Worker 経由)。インストール不要・サーバープロセス不要。
- **ローカルサーバー (FastAPI)** — 従来のクライアント/サーバー構成。開発・大容量データ・CuPy による GPU 加速に向く。

どちらのモードも同じ UI とノードカタログを駆動します。

## 特徴

- **100+ ノード** — 投影、FFT (2D/3D 自動判定)、フィルタ、マスク、整列 (`align.ap` が multi-ref / single-ref / rotation-only / class assignment を 1 本化)、逆投影 (`bp3f` / `bp3d` / `bp3g` / `bp3n`)、CTF、反復ループ (`flow.do` + `flow.state`)
- **フィードバック付き DO ループ** — 反復整列、クラス平均、SGD 的な再構成がそのまま書ける
- **ライブプレビュー** — すべてのノードに PNG サムネイル、スタックはフィルムストリップ、ボリュームは Three.js で回転可、Fourier は HSV で振幅を色相表示
- **チャレンジ** — 8 問の cryo-EM パズル (シフト復元、粒子検出、クラス平均、角度アサイン…)、NCC / クラスタリング一致率で自動判定
- **インタラクティブチュートリアル** — モバイルゲーム風ハイライト付きの 17 章 (基礎 1–12、比較 1–5)

## クイックスタート — Web ビルド (推奨)

必要環境: 直近 3 年くらいの最新ブラウザ (Chrome / Edge / Safari / Firefox)。ユーザー側に Python / pip / Node / numpy のインストール不要。

```sh
# ビルド済みリリースから:
cd frontend/dist
python3 -m http.server 8080   # 任意の静的サーバーで OK
open http://127.0.0.1:8080/

# ソースから:
cd frontend
npm install
npm run build
bash ../pyodide_poc/build_bundle.sh   # backend + pyspider を dist/bundle.zip に同梱
cd dist && python3 -m http.server 8080
```

初回ロードで Pyodide + numpy / scipy / Pillow / scikit-image (~45 MB) をダウンロード。以降はキャッシュされ、すべてブラウザ内で実行されます。

Windows / macOS の手順は [INSTALL.ja.md](INSTALL.ja.md) を参照してください。

## クイックスタート — ローカルサーバー

必要環境: Python 3.11+, Node 20+。

```sh
# バックエンド (port 8000)
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# フロントエンド (port 5173、/api は 8000 にプロキシ)
cd frontend
npm install
npm run dev
```

http://127.0.0.1:5173/ を開きます。

## ディレクトリ構成

```
backend/              FastAPI + プラグイン + DAG エンジン
  engine.py           トポロジ順実行 + DO ループ + ストリーミング
  nodes.py            基本数値カーネル
  plugins/builtin/    1 ノード = 1 ファイル (入れる/消すで有効化/無効化)
  tutorials/          インタラクティブチュートリアル (JSON)
  challenges/         8 問の cryo-EM パズル (JSON)
  api_shim.py         Pyodide 版エントリポイント (FastAPI と同じ API)
  bundled_data/       β-gal 参照マップ (MRC)
frontend/
  src/                React + React Flow の UI
    pyodideEngine.ts  Pyodide Web Worker クライアント
  public/
    pyodideWorker.js  Pyodide Web Worker 本体 (numpy/scipy ロード + エンジン実行)
  dist/               静的サイト (npm run build 後)
pyspider/             純 numpy の SPIDER カーネル群 (両モードで共有)
pyodide_poc/
  build_bundle.sh     backend + pyspider を bundle.zip に固める
```

## チャレンジ

| # | タイトル | 評価指標 | 合格ライン |
|---|---|---|---|
| 1 | シフトを戻す | NCC | 0.9 |
| 2 | 回転を戻す | NCC | 0.9 |
| 3 | 電顕写真から粒子を自動選択 | NCC | 0.5 |
| 4 | 実空間で投影像を作成 (FFT 不可) | NCC | 0.9 |
| 5 | 投影-断面定理で投影像作成 | NCC | 0.99 |
| 6 | ノイジースタックから K クラス平均を復元 | クラスタリング一致率 | 0.80 |
| 7 | ノイジースタックからクリーン平均を復元 | rot-NCC | 0.85 |
| 8 | 粒子像の配向をアサイン | NCC | 0.9 |

各チャレンジはパレットを必要ノードだけに制限し、ヒントに想定パイプラインを明記しています。

## コントリビュート

ノードプラグインは `backend/plugins/builtin/<カテゴリ>/<名前>.py` に置きます。1 ファイル 1 ノード。既存の兄弟ファイルをコピーしてテンプレートにし、`register(N(id, label, category, inputs, outputs, [params], doc, fn))` で登録。バックエンド再起動 (Pyodide モードならブラウザ再読み込みのみ) でパレットに出ます。

## ライセンス

MIT。[LICENSE](LICENSE) を参照。

## 謝辞

- **SPIDER** (Frank et al., Wadsworth) — アルゴリズムと 40 年にわたる cryo-EM 道具群に。
- **RELION** コンソーシアム — 付属 β-gal データセットのチュートリアル版を利用させていただきました。
- **React Flow** — UI のキャンバス / ノード / エッジライブラリ。
- **Pyodide** — ブラウザ内で CPython + numpy + scipy スタックを丸ごと動かせる魔法。
