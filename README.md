# TW Stock News Leaderboards (Rules-based, Fast)

本專案在 GitHub Actions 上每小時執行一次：
1. 讀取多個 RSS（`config/rss_sources.txt` 或環境變數 `RSS_URL`）。
2. 動態取得台灣上市/上櫃公司清單（多端點容錯）。
3. 使用 **FlashText** 關鍵字抽取與**距離窗** + **否定規則**的情緒打分（不使用大型模型，輕量高效）。
4. 按 1/3/5/10/30/60 天視窗輸出排行榜（CSV 與 Markdown）。

## 使用
- Fork / 新建 repo，放入此專案。
- 調整 `config/rss_sources.txt` 來源（可留 Google News）。
- 推送後，GitHub Actions 會每小時自動跑；或在 Actions 頁面手動觸發。

## 產物
- `data/news_log.jsonl`：累積打分紀錄。
- `outputs/leaderboard_{d}d.csv`：各視窗 CSV。
- `outputs/leaderboards.md`：總表。

## 效能與準確度
- 公司匹配採 FlashText，複雜度近似 O(L)。
- 僅在**同句**且公司名與情緒詞**距離 <= 24 字元**時記分。
- 否定詞（如「不」「未」「無」「非」）在情緒詞**前 3 字元**會**反轉極性**。
- 每則新聞每家公司**分數上限 |score| ≤ 2**，避免單篇刷分。

## 可調參數
見 `scripts/keywords.py` 與 `scripts/fetch_and_rank.py` 冒頭常數。

## 語法樹與語境加權（規則版）
- 解析 **讓步**（雖/儘管/縱然… + 但是/然而/仍/依然）與 **條件**（若/如果/只要…則/就/仍/反而）結構。
- 子句加權：讓步左 0.5、轉折右 1.1；條件前件 0.7、後件 1.1。

## 關聯傳播（母子公司 / 供應鏈）
- 讀取 `config/relations.json` 與 `relation_weights.json`。
- 以可配置權重將分數從公司向 **母公司/子公司/供應商/客戶** 傳播。
- 預設空表，避免誤配；建議逐步擴充。

## 來源權重自動學習 & 半衰期尋優
- `scripts/optimize.py`：半衰期網格（3,5,7,10,14,21,30）+ 來源權重坐標搜尋（0.8/1.0/1.2/1.4）。
- 目標：以 Top-5、隔日持有的 Sharpe 為目標。
- 輸出到 `outputs/optimize_report.md` 並更新 `config/*.json`。

## 風控回測（增強版）
- 參數：`config/risk.json`。
- 支援 **交易成本/滑價**、**多日持有**、**持倉上限**、**最大回撤停交易**。
- 產出 `outputs/backtest_enhanced.csv` 與 `outputs/backtest_enhanced_report.md`。

