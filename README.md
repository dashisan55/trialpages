# 🚉 ホームドア異常予兆検知システム

五反田駅ホームドアのモータ電流データを機械学習で解析し、
異常の予兆を自動検知するシステムです。

## システム構成

| 機能 | 内容 |
|------|------|
| 特徴量抽出 | Excelファイル → CSV変換 |
| モデル学習 | Isolation Forest / LOF / One-Class SVM |
| 自動再学習 | 毎週月曜 AM2:00 (GitHub Actions) |
| 朝の点検判定 | 毎日 AM5:30 (GitHub Actions) |
| 通知 | メール / Microsoft Teams |
| ダッシュボード | GitHub Pages |

## フォルダ構成

```
homedoor-anomaly-detection/
├── .github/
│   └── workflows/
│       ├── daily_train.yml        # 毎週自動再学習
│       └── morning_check.yml      # 毎朝自動判定
├── data/
│   ├── raw/                       # ← Excelファイルをここに置く
│   ├── features/                  # 自動生成される特徴量CSV
│   └── results/                   # 自動生成される判定結果JSON
├── models/                        # 自動生成される学習済みモデル
├── reports/                       # 自動生成されるレポート
├── src/
│   ├── extract_features.py        # 特徴量抽出
│   ├── train_models.py            # モデル学習・比較
│   ├── predict.py                 # 判定実行
│   ├── notify.py                  # 通知（メール・Teams）
│   └── validate_ai.py             # 検証AI・自動デバッグ
├── web/
│   └── index.html                 # GitHub Pages ダッシュボード
├── config/
│   └── settings.yaml              # 設定ファイル
├── requirements.txt
└── README.md
```

## セットアップ手順

### 1. ファイルをGitHubリポジトリに配置する

以下の順番でファイルを作成してください。

```
1.  config/settings.yaml
2.  requirements.txt
3.  .gitignore
4.  src/extract_features.py
5.  src/train_models.py
6.  src/predict.py
7.  src/notify.py
8.  src/validate_ai.py
9.  .github/workflows/daily_train.yml
10. .github/workflows/morning_check.yml
11. web/index.html
12. README.md
13. data/raw/.gitkeep
14. data/features/.gitkeep
15. data/results/.gitkeep
16. models/.gitkeep
17. reports/.gitkeep
```

### 2. GitHub Secretsを登録する

リポジトリの Settings → Secrets and variables → Actions から
以下の4つを登録してください。

| Name | 内容 |
|------|------|
| MAIL_FROM | 送信元Gmailアドレス |
| MAIL_TO | 送信先メールアドレス |
| MAIL_PASSWORD | GmailのアプリパスワードS（16桁） |
| TEAMS_WEBHOOK_URL | TeamsのIncoming Webhook URL |

### 3. GitHub Pagesを有効化する

Settings → Pages → Branch を「main」、
フォルダを「/web」に設定して Save。

### 4. Actionsをテスト実行する

Actions タブ → 「毎週自動再学習」→「Run workflow」

## データの置き方

`data/raw/` フォルダに以下の形式でExcelファイルを配置：

```
data/raw/
└── 20260630001027_五反田_1番線_1号車_1番ドア_閉動作.xlsx
```

## 異常判定のしきい値

| スコア | 判定 |
|--------|------|
| 0.00 〜 0.39 | ✅ 正常 |
| 0.40 〜 0.64 | ⚠️ 要注意 |
| 0.65 〜 1.00 | 🚨 異常予兆 |

## 使用モデル

| モデル | 特徴 |
|--------|------|
| Isolation Forest | 外れ値検出に強い・高速 |
| LOF | 局所的な密度異常を検出 |
| One-Class SVM | 非線形な境界を学習 |

個別モデル（ドアごと）と統合モデルを両方学習し、
精度比較レポートを自動生成します。

## ライセンス

社内利用限定
