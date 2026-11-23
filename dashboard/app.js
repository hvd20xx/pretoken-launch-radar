(function () {
  'use strict';

  const DATA = window.__RADAR_DATA__;
  const root = document.getElementById('app');
  if (!DATA || !root) return;

  const qs = new URLSearchParams(window.location.search);
  const state = {
    tier: qs.get('tier') || 'all',       // all | watchlist | early
    verdict: qs.get('verdict') || 'all', // all | FARM | MONITOR | PASS
    category: qs.get('category') || 'all',
    chain: qs.get('chain') || 'all',
    search: qs.get('q') || '',
    sort: qs.get('sort') || 'rank',      // rank | tvl | chg | age | odds | ev_hr
    dir: qs.get('dir') || 'asc',
    selectedSlug: qs.get('detail') || '',
    showLive: qs.get('live') === '1',
    limit: Number(qs.get('limit')) || 40,
  };

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fmtUsd(n) {
    if (n === null || n === undefined || !Number.isFinite(n)) return '—';
    const abs = Math.abs(n);
    const sign = n < 0 ? '-' : '';
    if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
    return `${sign}$${abs.toFixed(0)}`;
  }

  function fmtSignedUsd(n) {
    return `${n >= 0 ? '+' : '-'}$${Math.abs(n).toFixed(2)}`;
  }

  function fmtPct(n) {
    if (n === null || n === undefined) return '<span style="color:var(--muted)">n/a</span>';
    const cls = n >= 0 ? 'pos' : 'neg';
    return `<span class="${cls}">${n >= 0 ? '+' : ''}${n.toFixed(1)}%</span>`;
  }

  function fmtAge(days) {
    if (days === null || days === undefined) return 'n/a';
    if (days >= 365) return `${(days / 365).toFixed(1)}y`;
    return `${days}d`;
  }

  function rangeBarSvg(low, mid, high, maxHigh) {
    const logMin = Math.log10(5e5);
    const logMax = Math.log10(Math.max(maxHigh, 1e9));
    const xOf = (v) => {
      const t = (Math.log10(Math.max(v, 5e5)) - logMin) / Math.max(0.01, logMax - logMin);
      return Math.max(2, Math.min(108, 2 + t * 106));
    };
    const x1 = xOf(low);
    const xm = xOf(mid);
    const x2 = xOf(high);
    return `
      <svg width="110" height="14" viewBox="0 0 110 14" aria-label="80% conformal range">
        <line x1="2" y1="7" x2="108" y2="7" stroke="#DDD6C8" stroke-width="2"/>
        <line x1="${x1.toFixed(1)}" y1="7" x2="${x2.toFixed(1)}" y2="7" stroke="#1A5C73" stroke-width="4" stroke-linecap="round"/>
        <circle cx="${xm.toFixed(1)}" cy="7" r="3.4" fill="#0E6B4B" stroke="#FFFDF9" stroke-width="1"/>
      </svg>`;
  }

  const allCategories = [...new Set(DATA.rows.map((r) => r.category))].sort();
  const chainFreq = new Map();
  DATA.rows.forEach((r) => r.chains.forEach((c) => chainFreq.set(c, (chainFreq.get(c) || 0) + 1)));
  const topChains = [...chainFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18).map((x) => x[0]);
  const maxHigh = Math.max(...DATA.rows.map((r) => r.conformal_high_usd));

  function getFilteredRows() {
    const q = state.search.trim().toLowerCase();
    const out = DATA.rows.filter((r) => {
      if (state.tier !== 'all' && r.tier !== state.tier) return false;
      if (state.verdict !== 'all' && r.farm.verdict !== state.verdict) return false;
      if (state.category !== 'all' && r.category !== state.category) return false;
      if (state.chain !== 'all' && !r.chains.includes(state.chain)) return false;
      if (q && !r.name.toLowerCase().includes(q) && !r.slug.includes(q) && !r.category.toLowerCase().includes(q)) {
        return false;
      }
      return true;
    });

    const mult = state.dir === 'asc' ? 1 : -1;
    out.sort((a, b) => {
      let va = 0;
      let vb = 0;
      if (state.sort === 'rank') { va = a.rank; vb = b.rank; }
      else if (state.sort === 'tvl') { va = a.tvl_usd; vb = b.tvl_usd; }
      else if (state.sort === 'chg') { va = a.change_7d ?? -999; vb = b.change_7d ?? -999; }
      else if (state.sort === 'age') { va = a.age_days ?? -1; vb = b.age_days ?? -1; }
      else if (state.sort === 'odds') { va = a.peer_token_odds; vb = b.peer_token_odds; }
      else if (state.sort === 'ev_hr') { va = a.farm.ev_per_hour_usd; vb = b.farm.ev_per_hour_usd; }
      else { va = a.implied_value_usd; vb = b.implied_value_usd; }
      return (va - vb) * mult || a.rank - b.rank;
    });
    return out;
  }

  function renderDetail(row) {
    if (!row) return '';
    const vCls = row.farm.verdict.toLowerCase();
    const secCls = row.farm.security_tier === 'LOW' ? 'sec-low' : row.farm.security_tier === 'MODERATE' ? 'sec-mod' : 'sec-high';
    const chainsText = row.chains.slice(0, 6).join(', ') + (row.chains.length > 6 ? ` +${row.chains.length - 6}` : '');

    return `
      <aside class="detail" aria-label="Protocol detail inspector">
        <div class="detail-top">
          <div>
            <div style="display:flex;gap:5px;align-items:center;margin-bottom:4px;">
              <span class="badge ${row.tier === 'watchlist' ? 'wl' : 'early'}">#${row.rank} · ${row.tier === 'watchlist' ? 'WATCHLIST ≥$50M' : 'EARLY DISCOVERY'}</span>
              <span class="badge ${vCls}">${row.farm.verdict}</span>
              <span class="badge ${secCls}">AUDIT: ${row.farm.security_tier}</span>
            </div>
            <h2>${esc(row.name)}</h2>
            <div class="proto-sub">${esc(row.category)} · ${esc(chainsText)} · Tracked ${fmtAge(row.age_days)} (${esc(row.listed_at || 'undated')})</div>
          </div>
          <button class="close-btn" data-action="close-detail">Close ×</button>
        </div>

        <div class="detail-metrics">
          <div class="dm">
            <div class="dm-l">Implied Value (p50)</div>
            <div class="dm-v">${fmtUsd(row.implied_value_usd)}</div>
            <div class="dm-s">${row.implied_multiple.toFixed(2)}× TVL (${fmtUsd(row.tvl_usd)})</div>
          </div>
          <div class="dm">
            <div class="dm-l">80% Conformal Range</div>
            <div class="dm-v">${fmtUsd(row.conformal_low_usd)}–${fmtUsd(row.conformal_high_usd)}</div>
            <div class="dm-s">Empirical LOO p10–p90 band</div>
          </div>
          <div class="dm">
            <div class="dm-l">Peer Token Odds</div>
            <div class="dm-v">${Math.round(row.peer_token_odds * 100)}%</div>
            <div class="dm-s">${row.peer_tokened_count}/${row.peer_total_count} ${esc(row.category)} peers · EV ${fmtUsd(row.expected_launch_value_usd)}</div>
          </div>
        </div>

        <div class="section-h">Valuation Decomposition (Dual-CV Selected Model)</div>
        <div class="decomp-grid">
          <div class="dec-box"><span>TVL Power Base</span>${fmtUsd(row.factor_breakdown.base_usd)}</div>
          <div class="dec-box"><span>7d Momentum</span>×${row.factor_breakdown.momentum_mult.toFixed(2)} (${row.change_7d === null ? 'n/a' : `${row.change_7d >= 0 ? '+' : ''}${row.change_7d.toFixed(1)}%`})</div>
          <div class="dec-box"><span>Chain Breadth</span>×${row.factor_breakdown.chains_mult.toFixed(2)} (${row.chains.length} ch.)</div>
          <div class="dec-box"><span>Tenure Term</span>${row.factor_breakdown.age_mult === 1 ? 'Dropped (WF CV)' : `×${row.factor_breakdown.age_mult.toFixed(2)}`}</div>
        </div>

        <div class="section-h">Costed Farm Verdict &amp; Execution Economics</div>
        <div class="farm-econ ${vCls}">
          <div class="fe-item"><span>Verdict / Chain</span><strong>${row.farm.verdict} · ${esc(row.farm.route_chain)}</strong></div>
          <div class="fe-item"><span>Min Capital</span><strong>${fmtUsd(row.farm.min_capital_usd)}</strong></div>
          <div class="fe-item"><span>Est. Gas / Time</span><strong>$${row.farm.est_gas_usd.toFixed(2)} / ${row.farm.est_hours.toFixed(1)}h</strong></div>
          <div class="fe-item"><span>Net EV / Hour</span><strong>${fmtSignedUsd(row.farm.ev_per_hour_usd)}/hr</strong></div>
        </div>

        <div class="section-h">Actionable Farm Plan &amp; Pre-Deposit Security Audit (${row.farm_steps.length} Steps)</div>
        <ol class="steps-list">
          ${row.farm_steps.map((s, idx) => `<li><span class="step-num">${idx + 1}.</span><span>${esc(s)}</span></li>`).join('')}
        </ol>

        <div class="section-h">Closest Live-Tokened Peers in Snapshot (Deduplicated by gecko_id)</div>
        <table class="comps-table">
          <thead><tr><th>Protocol</th><th>Ticker</th><th>Category</th><th class="num">TVL</th><th class="num">MCap</th><th class="num">M/T</th></tr></thead>
          <tbody>
            ${row.comps.map((c) => `
              <tr>
                <td><strong>${esc(c.name)}</strong></td>
                <td style="font-family:var(--mono)">${esc(c.symbol)}</td>
                <td>${esc(c.category)}</td>
                <td class="num">${fmtUsd(c.tvl_usd)}</td>
                <td class="num">${fmtUsd(c.mcap_usd)}</td>
                <td class="num">${c.ratio.toFixed(2)}×</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </aside>`;
  }

  function render() {
    const filtered = getFilteredRows();
    const visible = filtered.slice(0, state.limit);
    const activeDetail = state.selectedSlug ? DATA.rows.find((r) => r.slug === state.selectedSlug) : null;

    const c = DATA.counts;
    const m = DATA.model;
    const farmCount = filtered.filter((r) => r.farm.verdict === 'FARM').length;
    const totalImplied = filtered.reduce((s, r) => s + r.implied_value_usd, 0);
    const topEvRow = [...filtered].sort((a, b) => b.farm.ev_per_hour_usd - a.farm.ev_per_hour_usd)[0];

    const pWl = ((c.watchlist / c.total_protocols) * 100).toFixed(1);
    const pEarly = ((c.early_discovery / c.total_protocols) * 100).toFixed(1);
    const pLive = ((c.live_tokened / c.total_protocols) * 100).toFixed(1);
    const pExcl = ((c.excluded_wrappers / c.total_protocols) * 100).toFixed(1);

    const sortHdr = (key, label, isNum) => {
      const cls = `${isNum ? 'num' : ''} ${state.sort === key ? 'sorted' : ''}`.trim();
      const arrow = state.sort === key ? (state.dir === 'asc' ? ' ↑' : ' ↓') : '';
      return `<th class="${cls}" data-sort="${key}">${label}${arrow}</th>`;
    };

    root.setAttribute('aria-busy', 'false');
    root.innerHTML = `
      <header class="mast">
        <div>
          <div class="mast-kicker">PRE-TOKEN LAUNCH RADAR · DUAL-CV ECONOMETRIC LEDGER</div>
          <h1>Pre-Token Protocols Ranked by Implied Launch Value</h1>
          <div class="mast-sub">Records no token on DefiLlama (not proof no token exists) · Every term fitted on ${c.calibration_issuers} deduplicated live-token issuers and selected via Leave-One-Out + Walk-Forward CV</div>
        </div>
        <div class="mast-meta">
          <span class="chip dark">SNAPSHOT ${esc(DATA.as_of)}</span>
          <span class="chip">FLOOR ${fmtUsd(DATA.floor_usd)}</span>
          <span class="chip">WATCHLIST ${c.watchlist}/${c.watchlist} COVERED</span>
          <span class="chip">MODEL: dashboard/model.mjs (0 TRUNCATED)</span>
        </div>
      </header>

      <section class="kpi-grid">
        <div class="kpi emerald">
          <div class="kpi-label">Scored Pre-Token Cohort</div>
          <div class="kpi-val">${c.scored_pretoken} <span style="font-size:12px;color:var(--muted)">(${c.watchlist} WL + ${c.early_discovery} Early)</span></div>
          <div class="kpi-sub">Active view: ${filtered.length} protocols · ${fmtUsd(totalImplied)} combined implied</div>
        </div>
        <div class="kpi teal">
          <div class="kpi-label">Dual Held-Out Validation</div>
          <div class="kpi-val">LOO ×${m.loo_mdme.toFixed(2)} · WF ×${m.wf_mdme.toFixed(2)}</div>
          <div class="kpi-sub">Median multiplicative error across ${c.calibration_issuers} issuers (${c.dated_calibration_issuers} dated)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">80% Conformal Range Band</div>
          <div class="kpi-val">×${m.conformal_p10_mult.toFixed(2)} to ×${m.conformal_p90_mult.toFixed(2)}</div>
          <div class="kpi-sub">Calibrated directly from signed out-of-sample LOO residuals</div>
        </div>
        <div class="kpi amber">
          <div class="kpi-label">Costed Farm Verdicts (Filtered)</div>
          <div class="kpi-val">${farmCount} FARM <span style="font-size:12px;color:var(--muted)">(top ${topEvRow ? fmtSignedUsd(topEvRow.farm.ev_per_hour_usd) + '/hr' : 'n/a'})</span></div>
          <div class="kpi-sub">Net of chain gas, 90d opp. cost &amp; pre-deposit security audit</div>
        </div>
      </section>

      <section class="integrity">
        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">Snapshot Integrity Funnel (${c.total_protocols} DefiLlama Protocols)</span>
            <span style="font-family:var(--mono);font-size:10.5px;color:var(--muted)">Zero hardcoded names</span>
          </div>
          <div class="funnel-bar">
            <div class="f-wl" style="width:${pWl}%"></div>
            <div class="f-early" style="width:${pEarly}%"></div>
            <div class="f-live" style="width:${pLive}%"></div>
            <div class="f-excl" style="width:${pExcl}%"></div>
          </div>
          <div class="funnel-legend">
            <span><i class="dot" style="background:var(--emerald)"></i><strong>${c.watchlist}</strong> Watchlist ≥$50M</span>
            <span><i class="dot" style="background:var(--teal)"></i><strong>${c.early_discovery}</strong> Early &lt;$50M</span>
            <span><i class="dot" style="background:#8E9AA7"></i><strong>${c.live_tokened}</strong> Live Token Set Aside</span>
            <span><i class="dot" style="background:#D2C8B8"></i><strong>${c.excluded_wrappers}</strong> Wrappers/Bridges/CEX Excluded</span>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">Model Spec &amp; Dual-CV Feature Ablation (Huber IRLS)</span>
            <span style="font-family:var(--mono);font-size:10.5px;color:var(--emerald);font-weight:700">Selected: ${esc(m.selected_label)}</span>
          </div>
          <div class="formula-line">${esc(m.formula_text)}</div>
          <div class="ablation-mini">
            ${m.ablation.map((a) => `<span class="abl-tag ${a.selected ? 'win' : ''}">${a.selected ? '★ ' : ''}${esc(a.id)}: LOO ×${a.loo_mdme} / WF ×${a.wf_mdme}</span>`).join('')}
          </div>
        </div>
      </section>

      <section class="toolbar">
        <div class="toolbar-group">
          <div class="seg" role="group" aria-label="Cohort filter">
            <button data-tier="all" class="${state.tier === 'all' ? 'active' : ''}">All Pre-Token (${c.scored_pretoken})</button>
            <button data-tier="watchlist" class="${state.tier === 'watchlist' ? 'active' : ''}">Watchlist ≥$50M (${c.watchlist})</button>
            <button data-tier="early" class="${state.tier === 'early' ? 'active' : ''}">Early Discovery (${c.early_discovery})</button>
          </div>
          <div class="seg" role="group" aria-label="Farm verdict filter">
            <button data-verdict="all" class="${state.verdict === 'all' ? 'active' : ''}">All Verdicts</button>
            <button data-verdict="FARM" class="${state.verdict === 'FARM' ? 'active' : ''}">FARM</button>
            <button data-verdict="MONITOR" class="${state.verdict === 'MONITOR' ? 'active' : ''}">MONITOR</button>
            <button data-verdict="PASS" class="${state.verdict === 'PASS' ? 'active' : ''}">PASS</button>
          </div>
        </div>
        <div class="toolbar-group">
          <select id="catSelect" aria-label="Category filter">
            <option value="all">All Categories (${allCategories.length})</option>
            ${allCategories.map((cat) => `<option value="${esc(cat)}" ${state.category === cat ? 'selected' : ''}>${esc(cat)}</option>`).join('')}
          </select>
          <select id="chainSelect" aria-label="Chain filter">
            <option value="all">All Chains</option>
            ${topChains.map((ch) => `<option value="${esc(ch)}" ${state.chain === ch ? 'selected' : ''}>${esc(ch)}</option>`).join('')}
          </select>
          <input id="searchBox" type="search" placeholder="Search protocol..." value="${esc(state.search)}" />
          <span class="count-pill">Showing ${visible.length} of ${filtered.length}</span>
        </div>
      </section>

      <section class="workspace ${activeDetail ? 'has-detail' : ''}">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                ${sortHdr('rank', '#', false)}
                <th>Protocol &amp; Category</th>
                ${sortHdr('rank', 'Implied (p50)', true)}
                <th class="num">80% Conformal Range</th>
                ${sortHdr('tvl', 'TVL', true)}
                ${sortHdr('chg', '7d Chg', true)}
                ${sortHdr('age', 'Age', true)}
                ${sortHdr('odds', 'Peer Odds', true)}
                ${sortHdr('ev_hr', 'Farm $/hr', true)}
                <th>Verdict &amp; Audit</th>
              </tr>
            </thead>
            <tbody>
              ${visible.map((r) => {
                const sel = activeDetail && activeDetail.slug === r.slug ? 'selected' : '';
                const vCls = r.farm.verdict.toLowerCase();
                const sCls = r.farm.security_tier === 'LOW' ? 'sec-low' : r.farm.security_tier === 'MODERATE' ? 'sec-mod' : 'sec-high';
                return `
                  <tr class="${sel}" data-slug="${esc(r.slug)}">
                    <td class="rank-cell">${r.rank}</td>
                    <td>
                      <div class="proto-name">
                        <span>${esc(r.name)}</span>
                        <span class="badge ${r.tier === 'watchlist' ? 'wl' : 'early'}">${r.tier === 'watchlist' ? 'WL' : 'EARLY'}</span>
                      </div>
                      <div class="proto-sub">${esc(r.category)} · ${esc(r.farm.route_chain)}${r.chains.length > 1 ? ` (+${r.chains.length - 1})` : ''}</div>
                    </td>
                    <td class="num"><strong>${fmtUsd(r.implied_value_usd)}</strong><div class="range-caption">${r.implied_multiple.toFixed(2)}× TVL</div></td>
                    <td class="range-cell">
                      ${rangeBarSvg(r.conformal_low_usd, r.implied_value_usd, r.conformal_high_usd, maxHigh)}
                      <div class="range-caption">${fmtUsd(r.conformal_low_usd)}–${fmtUsd(r.conformal_high_usd)}</div>
                    </td>
                    <td class="num">${fmtUsd(r.tvl_usd)}</td>
                    <td class="num">${fmtPct(r.change_7d)}</td>
                    <td class="num">${fmtAge(r.age_days)}</td>
                    <td class="num">${Math.round(r.peer_token_odds * 100)}%</td>
                    <td class="num ${r.farm.ev_per_hour_usd >= 12 ? 'pos' : r.farm.ev_per_hour_usd < 0 ? 'neg' : ''}">${fmtSignedUsd(r.farm.ev_per_hour_usd)}/h</td>
                    <td style="white-space:nowrap;">
                      <span class="badge ${vCls}">${r.farm.verdict}</span>
                      <span class="badge ${sCls}">${r.farm.security_tier}</span>
                    </td>
                  </tr>`;
              }).join('')}
            </tbody>
          </table>
          ${filtered.length > state.limit ? `<div style="padding:8px;text-align:center;border-top:1px solid var(--rule);"><button class="close-btn" data-action="load-more">Show all ${filtered.length} protocols</button></div>` : ''}
        </div>
        ${renderDetail(activeDetail)}
      </section>

      <footer class="footer-note">
        <span>Data: sealed snapshot (${esc(DATA.as_of)}) · ${c.live_tokened} protocols with trading tokens separated into calibration/reference cohort (top peer: ${esc(DATA.live_token_sample[0]?.name || 'n/a')} ${esc(DATA.live_token_sample[0]?.symbol || '')}).</span>
        <span>Implied values are statistical estimates, not guarantees of token issuance or financial advice.</span>
      </footer>
    `;

    bindEvents();
  }

  function bindEvents() {
    root.querySelectorAll('[data-tier]').forEach((b) =>
      b.addEventListener('click', () => {
        state.tier = b.getAttribute('data-tier');
        render();
      }),
    );
    root.querySelectorAll('[data-verdict]').forEach((b) =>
      b.addEventListener('click', () => {
        state.verdict = b.getAttribute('data-verdict');
        render();
      }),
    );
    const catSel = document.getElementById('catSelect');
    if (catSel) catSel.addEventListener('change', (e) => { state.category = e.target.value; render(); });
    const chSel = document.getElementById('chainSelect');
    if (chSel) chSel.addEventListener('change', (e) => { state.chain = e.target.value; render(); });
    const sBox = document.getElementById('searchBox');
    if (sBox) sBox.addEventListener('input', (e) => { state.search = e.target.value; render(); });

    root.querySelectorAll('thead th[data-sort]').forEach((th) =>
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort');
        if (state.sort === key) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        else {
          state.sort = key;
          state.dir = key === 'rank' ? 'asc' : 'desc';
        }
        render();
      }),
    );

    root.querySelectorAll('tbody tr[data-slug]').forEach((tr) =>
      tr.addEventListener('click', () => {
        state.selectedSlug = tr.getAttribute('data-slug');
        render();
      }),
    );

    const closeBtn = root.querySelector('[data-action="close-detail"]');
    if (closeBtn) closeBtn.addEventListener('click', () => { state.selectedSlug = ''; render(); });

    const moreBtn = root.querySelector('[data-action="load-more"]');
    if (moreBtn) moreBtn.addEventListener('click', () => { state.limit = 4000; render(); });
  }

  render();
})();
