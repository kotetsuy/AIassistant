# Phase 4 記録 — ドキュメント更新

- 実施日: 2026-08-13
- 結果: **完了**

## 更新したファイル

| ファイル | 内容 |
|---|---|
| `README.md` / `READMEJ.md` | 構成図の STT 表記、コンポーネント表(`Speech/` `ttllm/stt/` `bench/` 追加)、**STT バックエンド節を新設**、clone 手順に Speech を追加、symlink 手順に Speech を追加、venv 手順を共用 venv に全面改訂 |
| `TECHNICAL.md` / `TECHNICALJ.md` | エンドポイント表に `/chat_stream` `/transcribe_stream` `/stt_chat_stream` を追加、**STT バックエンド設計節を新設**(設計判断・ストリーミング経路・Phase 3 実測)、既知の制約に float32 固定とウォームアップ必須を追記、パス節を共用 venv に更新、トラブルシュートに 4 項目追加 |
| `ttllm/README.md` / `READMEJ.md` | タイトルを「STT ↔ llama.cpp bridge」に、構成図に `stt/` を追加、前提を共用 venv に、環境変数表を **STT 選択 / NeMo / whisperX の 3 表に再編**、フォールバック挙動の説明を追加 |
| `vtt/README.md` / `READMEJ.md` | 古い `WHISPERX_VENV` / `~/AIzunda` 参照を修正。vtt は ttllm と HTTP で話すだけなのでバックエンドの影響を受けない旨を明記 |

## 主な追記内容

### README — STT バックエンド節(新設)

`STT_BACKEND` の 3 モード表、切り替えコマンド、`/health` の確認方法。
「短い会話文では両者一致・NeMo が 2〜3 倍速い / 長く固有名詞の多い発話では whisperX が正確」
というトレードオフを明記した。**速度だけ見て選ばないように**するため。

ストリーミング STT は既定 off であることと、`?stt=stream` での有効化方法を記載。

### TECHNICAL — STT バックエンド設計節(新設)

- **なぜ 2 系統なのか**(ROCm 非公式サポート構成のリスク + 長文精度)
- `ttllm/stt/` の構成
- 設計判断: `model.transcribe()` を使わない理由(0.2〜0.4 秒の dataloader 再構築)、
  音声デコードを自前で持つ理由(ブラウザは webm/opus)、末尾パディングの理由、
  `ja-JP` が必須である点、フォールバックを必ず可視化する理由
- ストリーミング経路の図と、**MediaRecorder が使えない理由**、
  **mel を毎回全体から作り直す理由**(境界アーティファクトで「川は」→「かは」)
- Phase 3 の実測表

### トラブルシュートへの追加

移植で実際に踏んだ失敗が再現しうるものを項目化した。

| 現象 | 対処 |
|---|---|
| STT が急に whisperX になっている | `/health` の `stt.last_error` で理由を確認 |
| NeMo が offline mode で落ちる | `run.sh` は `HF_HUB_OFFLINE=1`。`install.sh` でモデルをキャッシュ |
| ストリーミングにしても部分転写が出ない | `stt.supports_streaming` を確認。whisperX では非対応 |
| 初回発話が遅い | warmup はロード + ウォームアップ推論で約 33 秒 |

## 残した記述

`README.md` / `TECHNICALJ.md` に残る `~/AIzunda` への言及は、
**「以前はここに置いていた」という経緯の説明**なので意図的に残した。

`three-vrm/README.md` にも古い `~/AIzunda` パスがあるが、今回のスコープ外
(STT 移植と無関係な箇所)なので触っていない。
