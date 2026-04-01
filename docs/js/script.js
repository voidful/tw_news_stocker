const LS_KEY_WL = "twnews_watchlist";

function getWatchlist() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY_WL) || "[]");
  } catch (e) {
    return [];
  }
}

function setWatchlist(arr) {
  localStorage.setItem(LS_KEY_WL, JSON.stringify(Array.from(new Set(arr))));
}

function addSparkline(el, series) {
  const c = document.createElement("canvas");
  c.width = 120;
  c.height = 28;
  c.style.marginLeft = "8px";
  el.appendChild(c);
  const xs = series.map(r => r.date);
  const ys = series.map(r => r.score);

  // Check if Chart is defined (it should be loaded via CDN)
  if (typeof Chart !== 'undefined') {
    new Chart(c.getContext("2d"), {
      type: "line",
      data: {
        labels: xs,
        datasets: [{
          data: ys,
          pointRadius: 0,
          tension: 0.3,
          borderColor: ys[ys.length - 1] >= 0 ? '#059669' : '#dc2626', // Color based on last value
          backgroundColor: 'transparent'
        }]
      },
      options: {
        responsive: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        },
        scales: {
          x: { display: false },
          y: { display: false }
        },
        elements: {
          line: { borderWidth: 1.5 }
        },
        animation: false
      }
    });
  }
}

async function main() {
  const daysSel = document.getElementById("range");
  const srcFreqEl = document.getElementById("srcFreq");
  const trustEl = document.getElementById("trustMask");
  const perStockN = document.getElementById("perStockN");
  const wlOnly = document.getElementById("wlOnly");
  const qEl = document.getElementById("q");
  const btnExport = document.getElementById("btnExport");
  const asofEl = document.getElementById("asof");

  // Show loading state
  asofEl.textContent = "載入資料中...";

  // 1. Load metadata and daily sentiment
  const [newsIndex, daily, sw] = await Promise.all([
    fetch("./data/news_index.json").then(r => r.json()).catch(() => []),
    fetch("./data/daily_sentiment.json").then(r => r.json()).catch(() => []),
    fetch("./data/source_weights.json").then(r => r.json()).catch(() => ({}))
  ]);

  if (!Array.isArray(newsIndex) || newsIndex.length === 0) {
    asofEl.textContent = "無法載入新聞索引。";
    return;
  }

  // 2. Determine which files to load based on selected range
  // Default range is 7 days
  let currentDays = +daysSel.value;

  async function loadNewsForRange(days) {
    const targetDates = newsIndex.slice(0, days + 2); // Fetch a bit more to be safe
    const requests = targetDates.map(date =>
      fetch(`./data/news/${date}.json`).then(r => r.json()).catch(() => [])
    );
    const results = await Promise.all(requests);
    return results.flat();
  }

  let news = await loadNewsForRange(currentDays);

  const wl = new Set(getWatchlist());

  // 來源頻數 Top-K
  function topHosts(k) {
    const cnt = {};
    news.forEach(n => {
      const h = n.source_host || (n.link ? new URL(n.link).host : "");
      cnt[h] = (cnt[h] || 0) + 1;
    });
    return Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, k).map(x => x[0]);
  }

  // 取得來源可信度權重
  function weightForHost(host, val) {
    const w = sw[host];
    if (w == null) return 1.0;
    if (typeof w === "number") return w;
    const sgn = (val >= 0) ? "pos" : "neg";
    return (w[sgn] != null) ? Number(w[sgn]) : 1.0;
  }

  function filterNews() {
    const days = +daysSel.value;
    const now = new Date();
    const mask = srcFreqEl.value;
    const allowHosts = mask === "all" ? null : new Set(topHosts(mask === "top5" ? 5 : 10));
    const q = (qEl.value || "").trim().toLowerCase();

    return news.filter(x => {
      if (now - new Date(x.ts) > days * 24 * 3600 * 1000) return false;
      const host = x.source_host || (x.link ? new URL(x.link).host : "");
      if (allowHosts && !allowHosts.has(host)) return false;
      if (wlOnly.checked) {
        const has = (x.codes || []).some(c => wl.has(c));
        if (!has) return false;
      }
      if (!q) return true;
      const inCodes = (x.codes || []).some(c => c.toLowerCase().includes(q));
      const inTitle = (x.title || "").toLowerCase().includes(q);
      return inCodes || inTitle;
    });
  }

  function seriesFor(code, days) {
    const end = new Date();
    const arr = [];
    for (let i = days - 1; i >= 0; --i) {
      const d = new Date(end - i * 24 * 3600 * 1000);
      const ds = d.toISOString().slice(0, 10);
      arr.push({ date: ds, score: 0 });
    }
    const sub = daily.filter(r => r.code === code);
    const idx = new Map(arr.map((r, i) => [r.date, i]));
    sub.forEach(r => {
      if (idx.has(r.date)) arr[idx.get(r.date)].score += +r.score || 0;
    });
    return arr;
  }

  function aggByCode(items) {
    const mode = trustEl.value; // off / pos / neg / both
    const m = {};
    items.forEach(n => {
      const host = n.source_host || (n.link ? new URL(n.link).host : "");
      let s = +n.sent_score || 0;
      if (mode !== "off" && s !== 0) {
        const applyPos = (mode === "pos" || mode === "both") && s > 0;
        const applyNeg = (mode === "neg" || mode === "both") && s < 0;
        if (applyPos || applyNeg) {
          s = s * weightForHost(host, s);
        }
      }
      (n.codes || []).forEach(c => {
        m[c] = (m[c] || 0) + s;
      });
    });
    return Object.entries(m).map(([code, score]) => ({ code, score }));
  }

  function recentNewsFor(code, items, k) {
    return items
      .filter(n => (n.codes || []).includes(code))
      .sort((a, b) => new Date(b.ts) - new Date(a.ts))
      .slice(0, k);
  }

  async function render() {
    // Check if we need to load more data
    const selectedDays = +daysSel.value;
    if (selectedDays > currentDays) {
      asofEl.textContent = "載入更多資料中...";
      news = await loadNewsForRange(selectedDays);
      currentDays = selectedDays;
    }

    const items = filterNews();
    const rows = aggByCode(items);
    const days = +daysSel.value;
    const kList = Math.max(1, Math.min(50, +perStockN.value || 5));

    const rows2 = wlOnly.checked ? rows.filter(r => wl.has(r.code)) : rows;

    const topp = rows2.filter(r => r.score >= 0).sort((a, b) => b.score - a.score).slice(0, 15);
    const topn = rows2.filter(r => r.score < 0).sort((a, b) => a.score - b.score).slice(0, 15);

    asofEl.textContent = `符合條件新聞 ${items.length} 則，股票數 ${rows2.length}` + (wlOnly.checked ? "（觀察名單）" : "");

    const fmt = v => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
    const toppEl = document.getElementById("toppos");
    toppEl.innerHTML = "";
    const topnEl = document.getElementById("topneg");
    topnEl.innerHTML = "";

    function mkRow(target, r) {
      const li = document.createElement("li");

      // Info container
      const info = document.createElement("div");
      info.style.display = "flex";
      info.style.alignItems = "center";
      info.style.flex = "1";

      const scoreBadge = document.createElement("span");
      scoreBadge.className = `badge ${r.score >= 0 ? "pos" : "neg"}`;
      scoreBadge.textContent = fmt(r.score);

      const codeTag = document.createElement("code");
      codeTag.className = "tag";
      codeTag.textContent = r.code;

      info.appendChild(scoreBadge);
      info.appendChild(codeTag);

      // Sparkline
      addSparkline(info, seriesFor(r.code, Math.max(7, days)));

      li.appendChild(info);

      // Actions container
      const actions = document.createElement("div");
      actions.style.marginLeft = "auto";
      actions.style.display = "flex";
      actions.style.alignItems = "center";

      const watch = document.createElement("button");
      watch.className = "secondary";
      const isOn = wl.has(r.code);
      watch.textContent = isOn ? "★ 取消" : "☆ 追蹤";
      watch.onclick = () => {
        if (wl.has(r.code)) wl.delete(r.code); else wl.add(r.code);
        setWatchlist(Array.from(wl));
        render();
      };

      const btnToggle = document.createElement("button");
      btnToggle.className = "linklike";
      btnToggle.style.marginLeft = "1rem";
      btnToggle.textContent = `展開 ${kList} 則`;

      actions.appendChild(watch);
      actions.appendChild(btnToggle);
      li.appendChild(actions);

      const detailWrap = document.createElement("div");
      detailWrap.className = "details";
      detailWrap.style.display = "none";
      detailWrap.style.flexBasis = "100%"; // Force new line

      btnToggle.onclick = () => {
        if (detailWrap.style.display === "none") {
          detailWrap.style.display = "block";
          btnToggle.textContent = "收合";
          const list = recentNewsFor(r.code, items, kList);
          detailWrap.innerHTML = list.map(n => {
            const s = +n.sent_score || 0;
            const badge = `<span class="badge ${s >= 0 ? "pos" : "neg"}">${fmt(s)}</span>`;
            const host = n.source_host || (n.link ? new URL(n.link).host : "");
            return `<div><small class="host">${n.ts} · ${host}</small><br>
              ${badge} <a href="${n.link || "#"}" target="_blank">${n.title || "(無標題)"}</a></div>`;
          }).join("");
        } else {
          detailWrap.style.display = "none";
          btnToggle.textContent = `展開 ${kList} 則`;
        }
      };

      li.appendChild(detailWrap);
      target.appendChild(li);
    }

    topp.forEach(r => mkRow(toppEl, r));
    topn.forEach(r => mkRow(topnEl, r));

    // 最新新聞
    const newsEl = document.getElementById("news");
    newsEl.innerHTML = "";
    const sorted = [...items].sort((a, b) => new Date(b.ts) - new Date(a.ts));
    sorted.slice(0, 300).forEach(n => {
      const s = +n.sent_score || 0;
      const badge = `<span class="badge ${s >= 0 ? "pos" : "neg"}">${fmt(s)}</span>`;
      const codes = (n.codes || []).map(c => `<code class="tag">${c}</code>`).join(" ");
      const host = n.source_host || (n.link ? new URL(n.link).host : "");
      const div = document.createElement("article");
      div.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <h5 style="margin-bottom:.3rem; flex:1;"><a href="${n.link || "#"}" target="_blank">${n.title || "(無標題)"}</a></h5>
            <small class="host" style="white-space:nowrap; margin-left:1rem;">${n.ts}</small>
        </div>
        <div style="margin-top:0.5rem;">
            <small class="host">${host}</small>
            <span style="margin:0 0.5rem;">|</span>
            ${badge} ${codes}
        </div>
      `;
      newsEl.appendChild(div);
    });
  }

  [srcFreqEl, trustEl, wlOnly].forEach(el => el.onchange = render);
  daysSel.onchange = render; // This will trigger data loading if needed
  perStockN.oninput = render;
  qEl.oninput = () => {
    clearTimeout(window._t);
    window._t = setTimeout(render, 200);
  };
  btnExport.onclick = () => {
    const items = filterNews();
    const rows = items.flatMap(n => {
      const s = +n.sent_score || 0;
      return (n.codes || [""]).map(c => ({
        ts: n.ts,
        code: c,
        score: s,
        title: n.title || "",
        link: n.link || "",
        host: n.source_host || ""
      }));
    });
    const header = "ts,code,score,title,link,host\n";
    const body = rows.map(r => [
      r.ts, r.code, r.score,
      `"${(r.title || "").replace(/"/g, '""')}"`,
      r.link, r.host
    ].join(",")).join("\n");
    const blob = new Blob([header + body], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "news_sentiment_export.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  render();
}

document.addEventListener("DOMContentLoaded", main);
