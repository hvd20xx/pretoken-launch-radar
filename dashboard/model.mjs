// Empirical pre-token launch valuation, triple-CV 9-spec feature keep/drop ablation
// (Leave-One-Out + Walk-Forward + Held-Out Category-Block CV), out-of-fold category
// empirical-Bayes shrinkage, adaptive 80% conformal bands, survival peer launch odds,
// and $/hr costed farm + Sybil dilution + security audit engine.

const EXCLUDED_CATEGORIES = new Set([
  'Bridge',
  'Canonical Bridge',
  'Liquid Staking',
  'Liquid Restaking',
  'Restaked BTC',
  'CEX',
]);

const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/i;
const VERSION_SUFFIX = / (?:v\d+(?:\.\d+)?|\d+\.\d+)$/i;
const WRAPPED_WORD = /\b(?:wrapped|staked|staking)\b/i;

const CHAIN_GAS_USD = {
  Solana: 0.08, Base: 0.35, Arbitrum: 0.45, Optimism: 0.35, Polygon: 0.25,
  Sui: 0.12, Aptos: 0.12, Avalanche: 0.65, BSC: 0.4, Binance: 0.4,
  Linea: 0.55, Scroll: 0.55, Sonic: 0.15, Hyperliquid: 0.2, 'Hyperliquid L1': 0.2,
  Starknet: 0.4, Monad: 0.25, Unichain: 0.3, Berachain: 0.35, Gnosis: 0.15,
  xDai: 0.15, Mantle: 0.3, Blast: 0.45, Mode: 0.35, Bitcoin: 12.0, Ethereum: 18.0,
};

export function parseSnapshot(raw) {
  const data = JSON.parse(raw);
  if (!data || !Array.isArray(data.protocols) || !Array.isArray(data.watchlist)) {
    throw new Error('Invalid snapshot.json structure');
  }
  return data;
}

function daysBetween(isoDate, asOfMs) {
  if (!isoDate) return null;
  const ms = Date.parse(`${isoDate}T00:00:00Z`);
  if (!Number.isFinite(ms)) return null;
  return Math.max(1, Math.round((asOfMs - ms) / 86_400_000));
}

function clampMomentum(chg) {
  if (chg === null || chg === undefined || !Number.isFinite(chg)) return 0;
  return Math.max(-0.5, Math.min(1.0, chg / 100));
}

function isWrapperOrExcluded(p) {
  const cat = p.category ?? '';
  return (
    EXCLUDED_CATEGORIES.has(cat) ||
    WRAPPED_WORD.test(p.name) ||
    EXCHANGE_PREFIX.test(p.name) ||
    VERSION_SUFFIX.test(p.name)
  );
}

function solveLinearSystem(A, b) {
  const m = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < m; c += 1) {
    let piv = c;
    for (let r = c + 1; r < m; r += 1) {
      if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    }
    const tmp = M[c];
    M[c] = M[piv];
    M[piv] = tmp;
    const diag = M[c][c];
    if (Math.abs(diag) < 1e-12) continue;
    for (let r = c + 1; r < m; r += 1) {
      const factor = M[r][c] / diag;
      for (let k = c; k <= m; k += 1) M[r][k] -= factor * M[c][k];
    }
  }
  const x = new Array(m).fill(0);
  for (let r = m - 1; r >= 0; r -= 1) {
    let s = M[r][m];
    for (let k = r + 1; k < m; k += 1) s -= M[r][k] * x[k];
    x[r] = Math.abs(M[r][r]) < 1e-12 ? 0 : s / M[r][r];
  }
  return x;
}

/** Deterministic IRLS Huber regression with L2 shrinkage on non-intercept slopes. */
export function fitHuber(X, y, ridge = 0.1, delta = 0.6) {
  const n = X.length;
  const m = X[0].length;
  let wts = new Array(n).fill(1);
  let w = new Array(m).fill(0);
  for (let iter = 0; iter < 16; iter += 1) {
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const b = new Array(m).fill(0);
    for (let i = 0; i < n; i += 1) {
      const wi = wts[i];
      const xi = X[i];
      const yi = y[i];
      for (let j = 0; j < m; j += 1) {
        b[j] += wi * xi[j] * yi;
        for (let k = 0; k < m; k += 1) A[j][k] += wi * xi[j] * xi[k];
      }
    }
    for (let j = 1; j < m; j += 1) A[j][j] += ridge;
    w = solveLinearSystem(A, b);
    wts = y.map((yi, i) => {
      let pred = 0;
      for (let j = 0; j < m; j += 1) pred += X[i][j] * w[j];
      const err = Math.abs(yi - pred);
      return err <= delta ? 1 : delta / err;
    });
  }
  return w;
}

function quantile(sorted, q) {
  if (sorted.length === 0) return 0;
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
}

/** Out-of-fold median empirical-Bayes category residual adjustments (shrinkage k=6). */
function fitCategoryShrinkage(trainRows, w, featureFn, kPrior = 6) {
  const buckets = new Map();
  for (const r of trainRows) {
    const x = featureFn(r);
    const pred = x.reduce((s, v, idx) => s + v * w[idx], 0);
    const list = buckets.get(r.cat) ?? [];
    list.push(r.y - pred);
    buckets.set(r.cat, list);
  }
  const adj = new Map();
  for (const [cat, list] of buckets.entries()) {
    list.sort((a, b) => a - b);
    const med = quantile(list, 0.5);
    adj.set(cat, (med * list.length) / (list.length + kPrior));
  }
  return adj;
}

/** Right-censoring-adjusted Bayes survival peer token-issuance odds on matured peers (age >= 365d). */
function fitSurvivalTokenOdds(maturePeers, tauMedDays, kPrior = 5) {
  const baseRate = maturePeers.reduce((s, r) => s + r.tok, 0) / Math.max(1, maturePeers.length);
  const catMap = new Map();
  for (const r of maturePeers) {
    const st = catMap.get(r.cat) ?? { t: 0, n: 0 };
    st.t += r.tok;
    st.n += 1;
    catMap.set(r.cat, st);
  }
  const catLifetime = new Map();
  for (const [c, st] of catMap.entries()) {
    catLifetime.set(c, (st.t + kPrior * baseRate) / (st.n + kPrior));
  }
  const predictLifetime = (cat) => catLifetime.get(cat) ?? baseRate;
  const predictConditional = (cat, ageDays) => {
    const pc = predictLifetime(cat);
    if (ageDays === null || ageDays <= 0) return pc;
    const surv = Math.exp((-0.35 * ageDays) / Math.max(180, tauMedDays));
    return (pc * surv) / Math.max(1e-6, 1 - pc * (1 - surv));
  };
  return { baseRate, predictLifetime, predictConditional };
}

const CANDIDATE_SPECS = [
  { id: 'tvl_only', label: 'TVL power-law baseline', terms: ['intercept', 'log10(TVL)'], catEb: false, features: (r) => [1, r.xTvl] },
  { id: 'tvl_mom', label: 'TVL + 7d growth', terms: ['intercept', 'log10(TVL)', 'chg_7d'], catEb: false, features: (r) => [1, r.xTvl, r.xChg] },
  { id: 'tvl_age', label: 'TVL + tenure', terms: ['intercept', 'log10(TVL)', 'log10(age+30)'], catEb: false, features: (r) => [1, r.xTvl, r.xAge] },
  { id: 'tvl_chains', label: 'TVL + chain breadth', terms: ['intercept', 'log10(TVL)', 'log10(chains)'], catEb: false, features: (r) => [1, r.xTvl, r.xChains] },
  { id: 'tvl_cat_eb', label: 'TVL + median category EB', terms: ['intercept', 'log10(TVL)', 'cat_med_eb(k=6)'], catEb: true, features: (r) => [1, r.xTvl] },
  { id: 'tvl_mom_chains', label: 'TVL + 7d growth + chain breadth', terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(chains)'], catEb: false, features: (r) => [1, r.xTvl, r.xChg, r.xChains] },
  { id: 'tvl_mom_chains_cat', label: 'TVL + 7d growth + chains + median category EB', terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(chains)', 'cat_med_eb(k=6)'], catEb: true, features: (r) => [1, r.xTvl, r.xChg, r.xChains] },
  { id: 'tvl_mom_age_chains', label: 'TVL + growth + tenure + chains', terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(age+30)', 'log10(chains)'], catEb: false, features: (r) => [1, r.xTvl, r.xChg, r.xAge, r.xChains] },
  { id: 'tvl_mom_age_chains_cat', label: 'Full 5-term (+ tenure + median category EB)', terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(age+30)', 'log10(chains)', 'cat_med_eb(k=6)'], catEb: true, features: (r) => [1, r.xTvl, r.xChg, r.xAge, r.xChains] },
];

function pickCheapestChain(chains) {
  if (!chains || chains.length === 0) return { chain: 'Ethereum', gas: 18.0 };
  let bestChain = chains[0];
  let bestGas = CHAIN_GAS_USD[bestChain] ?? 2.5;
  for (const c of chains) {
    const g = CHAIN_GAS_USD[c] ?? 2.5;
    if (g < bestGas) { bestGas = g; bestChain = c; }
  }
  return { chain: bestChain, gas: bestGas };
}

function fmtUsdShort(n) {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

/** Costed farm planner: chain gas, min Sybil-safe capital, 90d opp-cost, Sybil dilution, $/hr EV, and security audit. */
export function buildFarmPlan(p, impliedValue, launchOdds, ageDays) {
  const cat = p.category ?? 'DeFi';
  const { chain, gas } = pickCheapestChain(p.chains);
  const isSolana = p.chains.some((c) => c.toLowerCase() === 'solana');
  const isMainnetOnly = p.chains.length === 1 && p.chains[0] === 'Ethereum';

  const minCapital = isMainnetOnly
    ? p.tvl_usd >= 200_000_000 ? 5000 : 2500
    : p.tvl_usd >= 200_000_000 ? 2000 : p.tvl_usd >= 50_000_000 ? 1000 : 500;

  const estHours = cat === 'Dexs' || cat === 'Derivatives' || cat === 'Perps' ? 4.0 : isMainnetOnly ? 2.5 : 3.0;
  const totalGas = Number((gas * (estHours >= 4 ? 8 : 4)).toFixed(2));

  // Two-component transparent airdrop capture (7.5% community pool = 4.5% TVL pro-rata + 3.0% active-wallet tier)
  const proRataEv = (minCapital / Math.max(p.tvl_usd, 5_000_000)) * (impliedValue * 0.045);
  const estActiveWallets = Math.max(800, Math.round(Math.sqrt(Math.max(p.tvl_usd, 5_000_000)) * 1.4));
  const tierEv = (impliedValue * 0.03) / estActiveWallets;
  const grossEv = Math.min((proRataEv + tierEv) * launchOdds, minCapital * 0.35);

  const yieldBearing = ['Lending', 'CDP', 'Yield', 'Yield Aggregator', 'Risk Curators', 'Onchain Capital Allocator', 'RWA', 'Basis Trading'].includes(cat);
  const frictionRate = yieldBearing ? 0.0025 : (cat === 'Dexs' || cat === 'Derivatives' || cat === 'Perps') ? 0.0035 : 0.005;
  const oppCost = Number((minCapital * frictionRate).toFixed(2));
  const netEv = Math.round((grossEv - totalGas - oppCost) * 100) / 100;
  const evPerHour = Math.round((netEv / estHours) * 100) / 100;
  const annualizedRoiPct = Number((((netEv / minCapital) * (365 / 90)) * 100).toFixed(1));
  const dilutionRisk = p.tvl_usd >= 500_000_000 ? 'HIGH' : p.tvl_usd >= 100_000_000 ? 'MED' : 'LOW';

  const flags = [];
  if (p.change_7d !== null && p.change_7d <= -15) {
    flags.push(`Sharp 7d TVL outflow (${p.change_7d.toFixed(1)}%) — verify withdrawal liquidity before deposit`);
  }
  if (ageDays !== null && ageDays < 90) {
    flags.push(`Young protocol (${ageDays}d tracked) — cap exposure until first independent audit settles`);
  }
  if (cat === 'Risk Curators' || cat === 'Onchain Capital Allocator' || cat === 'Yield') {
    flags.push('Curator/vault admin role risk — inspect timelock delay and strategy whitelist permissions');
  }
  flags.push(
    isSolana
      ? 'Solana SPL check — verify program upgrade authority, freeze/mint flags & Token-2022 hooks'
      : 'EVM contract check — confirm verified explorer source, proxy admin multisig & withdraw path',
  );

  const securityTier =
    (p.change_7d !== null && p.change_7d <= -20) || (ageDays !== null && ageDays < 60)
      ? 'ELEVATED'
      : flags.length >= 2 ? 'MODERATE' : 'LOW';

  let verdict;
  if (evPerHour >= 12 && securityTier !== 'ELEVATED') verdict = 'FARM';
  else if (evPerHour >= 3 || (launchOdds >= 0.45 && p.tvl_usd >= 50_000_000 && netEv > 0)) verdict = 'MONITOR';
  else verdict = 'PASS';

  const catAction =
    cat === 'Dexs' || cat === 'Derivatives'
      ? `Execute 4–6 organic swaps or tight-range LP positions totaling ${fmtUsdShort(minCapital)} on ${chain} to qualify for both volume and liquidity snapshots`
      : cat === 'Lending' || cat === 'CDP'
        ? `Supply ${fmtUsdShort(minCapital)} in major collateral (ETH/USDC/BTC) on ${chain} and maintain a conservative health factor above 2.0 across weekly checkpoints`
        : cat === 'Risk Curators' || cat === 'Yield' || cat === 'Yield Aggregator' || cat === 'Basis Trading'
          ? `Deposit ${fmtUsdShort(minCapital)} into ${p.name}'s deepest ${chain} vault (avoiding illiquid side pools) and hold through at least 2 epoch settlements`
          : cat === 'RWA'
            ? `Complete KYC/allowlist onboarding if required and allocate ${fmtUsdShort(minCapital)} into ${p.name}'s primary ${chain} yield-bearing note for 60–90 days`
            : `Deploy ${fmtUsdShort(minCapital)} into ${p.name}'s core ${chain} contract and record on-chain interactions across at least 3 distinct calendar weeks`;

  const secStep = isSolana
    ? `Pre-deposit audit (${securityTier} risk): inspect ${p.name}'s Solana program upgrade authority, confirm no freeze/permanent-delegate Token-2022 hooks, and test a $20 withdraw before full size`
    : `Pre-deposit audit (${securityTier} risk): confirm verified contract source on ${chain}, check proxy admin multisig/timelock, and run a $25 deposit-and-withdraw dry run first`;

  const cadenceStep = `Execution budget: route via ${chain} (~$${totalGas.toFixed(2)} est. gas across 12 wks, ~${estHours.toFixed(1)}h time, $${oppCost.toFixed(0)} 90d friction, ${dilutionRisk} TVL dilution) — net EV ${netEv >= 0 ? '+' : '-'}$${Math.abs(netEv).toFixed(2)} (${evPerHour >= 0 ? '+' : '-'}$${Math.abs(evPerHour).toFixed(2)}/hr, ${annualizedRoiPct >= 0 ? '+' : ''}${annualizedRoiPct}% ann., ${verdict})`;

  const monitorStep =
    p.change_7d !== null && p.change_7d < -10
      ? `Risk trigger: ${p.name} saw ${p.change_7d.toFixed(1)}% 7d TVL drawdown; pause top-ups until 7d TVL stabilizes and exit immediately if TVL drops another 25%`
      : `Snapshot discipline: track ${p.name}'s ${fmtUsdShort(p.tvl_usd)} TVL and peer token odds (${Math.round(launchOdds * 100)}% in ${cat}, ~${estActiveWallets.toLocaleString('en-US')} est. active wallets); exit if TGE terms dilute small wallets`;

  const claimStep = `Position tracking: keep a dedicated farming wallet on ${chain}, log deposit tx hashes, review governance/points disclosures monthly, and revoke excess token approvals after exit`;

  return {
    farm: {
      verdict,
      route_chain: chain,
      min_capital_usd: minCapital,
      est_gas_usd: totalGas,
      est_hours: estHours,
      expected_ev_usd: netEv,
      ev_per_hour_usd: evPerHour,
      ann_roi_pct: annualizedRoiPct,
      dilution_risk: dilutionRisk,
      est_wallets: estActiveWallets,
      security_tier: securityTier,
      security_flags: flags,
    },
    steps: [secStep, catAction, cadenceStep, monitorStep, claimStep],
  };
}

/** Deduplicate multi-version live-token issuers by gecko_id, run triple-CV selection, and score pre-token protocols. */
export function buildRadar(snapshot) {
  const asOfMs = Date.parse(`${snapshot.as_of}T00:00:00Z`);
  const watchlistSet = new Set(snapshot.watchlist);

  const catStats = new Map();
  let liveTokenCount = 0;
  for (const p of snapshot.protocols) {
    const hasToken = p.gecko_id !== null || p.symbol !== null;
    if (hasToken) liveTokenCount += 1;
    const cat = p.category ?? 'Uncategorized';
    const st = catStats.get(cat) ?? { total: 0, tokened: 0 };
    st.total += 1;
    if (hasToken) st.tokened += 1;
    catStats.set(cat, st);
  }

  // Deduplicate live-token calibration cohort by gecko_id (summing multi-version TVL)
  const byGecko = new Map();
  for (const p of snapshot.protocols) {
    if (!p.gecko_id || p.tvl_usd <= 0) continue;
    const prev = byGecko.get(p.gecko_id);
    const listedMs = p.listed_at ? Date.parse(`${p.listed_at}T00:00:00Z`) : null;
    const excl = EXCLUDED_CATEGORIES.has(p.category ?? '');
    const mcap = p.mcap_usd !== null && p.mcap_usd > 0 ? p.mcap_usd : 0;
    if (!prev) {
      byGecko.set(p.gecko_id, {
        gecko_id: p.gecko_id,
        name: p.name.replace(VERSION_SUFFIX, ''),
        symbol: p.symbol ?? p.gecko_id.toUpperCase(),
        mcap_usd: mcap,
        tvl_usd: p.tvl_usd,
        wChg: p.change_7d !== null ? p.change_7d * p.tvl_usd : 0,
        chgW: p.change_7d !== null ? p.tvl_usd : 0,
        minListedMs: listedMs,
        chains: new Set(p.chains),
        cat: p.category ?? 'Uncategorized',
        allExcl: excl,
        maxPartTvl: p.tvl_usd,
      });
    } else {
      prev.tvl_usd += p.tvl_usd;
      if (mcap > prev.mcap_usd) prev.mcap_usd = mcap;
      if (p.change_7d !== null) {
        prev.wChg += p.change_7d * p.tvl_usd;
        prev.chgW += p.tvl_usd;
      }
      if (listedMs !== null && (prev.minListedMs === null || listedMs < prev.minListedMs)) prev.minListedMs = listedMs;
      for (const c of p.chains) prev.chains.add(c);
      if (!excl) prev.allExcl = false;
      if (p.tvl_usd > prev.maxPartTvl) {
        prev.maxPartTvl = p.tvl_usd;
        prev.cat = p.category ?? prev.cat;
      }
    }
  }

  const rawCalib = [...byGecko.values()].filter((g) => !g.allExcl && g.tvl_usd > 0 && g.mcap_usd > 0);
  const datedAges = rawCalib
    .filter((g) => g.minListedMs !== null)
    .map((g) => Math.log10(Math.max(1, (asOfMs - g.minListedMs) / 86_400_000) + 30));
  const meanLogAge = datedAges.length > 0 ? datedAges.reduce((a, b) => a + b, 0) / datedAges.length : 2.5;

  const calib = rawCalib
    .map((g) => ({
      gecko_id: g.gecko_id,
      name: g.name,
      symbol: g.symbol,
      cat: g.cat,
      tvl_usd: g.tvl_usd,
      mcap_usd: g.mcap_usd,
      y: Math.log10(g.mcap_usd),
      xTvl: Math.log10(g.tvl_usd),
      xChg: clampMomentum(g.chgW > 0 ? g.wChg / g.chgW : 0),
      xAge: g.minListedMs !== null ? Math.log10(Math.max(1, (asOfMs - g.minListedMs) / 86_400_000) + 30) : meanLogAge,
      xChains: Math.log10(Math.max(1, g.chains.size)),
      listedMs: g.minListedMs,
    }))
    .sort((a, b) => (a.gecko_id < b.gecko_id ? -1 : 1));

  // Triple Held-Out CV: (1) Leave-One-Out CV, (2) Chronological Walk-Forward CV, (3) Leave-One-Category-Out Block CV
  const dated = calib.filter((d) => d.listedMs !== null).sort((a, b) => a.listedMs - b.listedMs);
  const undated = calib.filter((d) => d.listedMs === null);
  const distinctCalibCats = [...new Set(calib.map((c) => c.cat))].sort();

  const ablationResults = CANDIDATE_SPECS.map((spec) => {
    const looResiduals = [];
    for (let i = 0; i < calib.length; i += 1) {
      const train = calib.filter((_, j) => j !== i);
      const w = fitHuber(train.map((r) => spec.features(r)), train.map((r) => r.y));
      const catAdj = spec.catEb ? fitCategoryShrinkage(train, w, spec.features) : null;
      const xi = spec.features(calib[i]);
      const pred = xi.reduce((s, v, idx) => s + v * w[idx], 0) + (catAdj ? (catAdj.get(calib[i].cat) ?? 0) : 0);
      looResiduals.push(calib[i].y - pred);
    }
    const absLoo = looResiduals.map((r) => Math.abs(r)).sort((a, b) => a - b);
    const looMdme = Math.pow(10, quantile(absLoo, 0.5));

    const wfResiduals = [];
    const minTrain = Math.min(20, Math.floor(dated.length / 2));
    for (let i = minTrain; i < dated.length; i += 1) {
      const train = [...undated, ...dated.slice(0, i)];
      const w = fitHuber(train.map((r) => spec.features(r)), train.map((r) => r.y));
      const catAdj = spec.catEb ? fitCategoryShrinkage(train, w, spec.features) : null;
      const xi = spec.features(dated[i]);
      const pred = xi.reduce((s, v, idx) => s + v * w[idx], 0) + (catAdj ? (catAdj.get(dated[i].cat) ?? 0) : 0);
      wfResiduals.push(dated[i].y - pred);
    }
    const wfAbs = wfResiduals.map((r) => Math.abs(r)).sort((a, b) => a - b);
    const wfMdme = Math.pow(10, quantile(wfAbs, 0.5));

    const catBlockResiduals = [];
    for (const heldCat of distinctCalibCats) {
      const train = calib.filter((r) => r.cat !== heldCat);
      const test = calib.filter((r) => r.cat === heldCat);
      if (train.length < 10 || test.length === 0) continue;
      const w = fitHuber(train.map((r) => spec.features(r)), train.map((r) => r.y));
      for (const ti of test) {
        const xi = spec.features(ti);
        const pred = xi.reduce((s, v, idx) => s + v * w[idx], 0);
        catBlockResiduals.push(ti.y - pred);
      }
    }
    const catAbs = catBlockResiduals.map((r) => Math.abs(r)).sort((a, b) => a - b);
    const catMdme = Math.pow(10, quantile(catAbs, 0.5));

    return { spec, loo: looMdme, wf: wfMdme, catCv: catMdme, geo: Math.sqrt(looMdme * wfMdme), looResiduals, wfResiduals, catBlockResiduals };
  });

  ablationResults.sort((a, b) => a.geo - b.geo);
  const winner = ablationResults[0];
  const finalWeights = fitHuber(calib.map((r) => winner.spec.features(r)), calib.map((r) => r.y));
  const finalCatAdj = winner.spec.catEb ? fitCategoryShrinkage(calib, finalWeights, winner.spec.features) : null;
  const weightMap = {};
  winner.spec.terms.forEach((t, idx) => {
    if (idx < finalWeights.length) weightMap[t] = Number(finalWeights[idx].toFixed(4));
  });

  // Global + Category-Adaptive EB 80% Conformal Intervals
  const sortedSignedRes = [...winner.looResiduals].sort((a, b) => a - b);
  const q10Log = quantile(sortedSignedRes, 0.1);
  const q90Log = quantile(sortedSignedRes, 0.9);
  const confLowMult = Math.pow(10, q10Log);
  const confHighMult = Math.pow(10, q90Log);
  const looCov80Pct = Number(((winner.looResiduals.filter((r) => r >= q10Log && r <= q90Log).length / Math.max(1, winner.looResiduals.length)) * 100).toFixed(1));
  const wfCov80Pct = Number(((winner.wfResiduals.filter((r) => r >= q10Log && r <= q90Log).length / Math.max(1, winner.wfResiduals.length)) * 100).toFixed(1));
  const catCov80Pct = Number(((winner.catBlockResiduals.filter((r) => r >= q10Log && r <= q90Log).length / Math.max(1, winner.catBlockResiduals.length)) * 100).toFixed(1));

  // Category dispersion shrinkage (widens conformal range in noisy categories, tightens in homogeneous ones, shrunk k=8 to global)
  const catResByCat = new Map();
  calib.forEach((c, idx) => {
    const arr = catResByCat.get(c.cat) ?? [];
    arr.push(Math.abs(winner.looResiduals[idx]));
    catResByCat.set(c.cat, arr);
  });
  const globalMad = quantile(calib.map((_, idx) => Math.abs(winner.looResiduals[idx])).sort((a, b) => a - b), 0.5) || 0.3;
  const catDispScale = new Map();
  for (const [cat, arr] of catResByCat.entries()) {
    arr.sort((a, b) => a - b);
    const localMad = quantile(arr, 0.5);
    const shrunkMad = (arr.length * localMad + 8 * globalMad) / (arr.length + 8);
    catDispScale.set(cat, Math.max(0.82, Math.min(1.28, shrunkMad / globalMad)));
  }

  const allPeerComps = calib
    .map((c) => ({
      gecko_id: c.gecko_id,
      name: c.name,
      symbol: c.symbol,
      category: c.cat,
      tvl_usd: Math.round(c.tvl_usd),
      mcap_usd: Math.round(c.mcap_usd),
      ratio: Number((c.mcap_usd / c.tvl_usd).toFixed(2)),
    }))
    .sort((a, b) => b.mcap_usd - a.mcap_usd);

  let excludedWrappers = 0;
  const targets = [];
  for (const p of snapshot.protocols) {
    if (watchlistSet.has(p.slug)) targets.push({ p, tier: 'watchlist' });
    else if (p.gecko_id === null && p.symbol === null) {
      if (isWrapperOrExcluded(p)) excludedWrappers += 1;
      else targets.push({ p, tier: 'early' });
    }
  }

  const datedCalibAges = rawCalib
    .filter((g) => g.minListedMs !== null)
    .map((g) => Math.max(1, Math.round((asOfMs - g.minListedMs) / 86_400_000)))
    .sort((a, b) => a - b);
  const tauMedDays = datedCalibAges.length > 0 ? quantile(datedCalibAges, 0.5) : 730;
  const maturePeers = snapshot.protocols
    .filter((p) => {
      const hasTok = p.gecko_id !== null || p.symbol !== null;
      if (!hasTok && !watchlistSet.has(p.slug) && isWrapperOrExcluded(p)) return false;
      if (hasTok && EXCLUDED_CATEGORIES.has(p.category ?? '')) return false;
      const ad = daysBetween(p.listed_at, asOfMs);
      return ad === null || ad >= 365;
    })
    .map((p) => ({ tok: p.gecko_id !== null || p.symbol !== null ? 1 : 0, cat: p.category ?? 'Uncategorized' }));
  const fullOddsModel = fitSurvivalTokenOdds(maturePeers, tauMedDays);
  const brierBase = maturePeers.reduce((s, r) => s + Math.pow(r.tok - fullOddsModel.baseRate, 2), 0) / Math.max(1, maturePeers.length);
  let brierLoo = 0;
  for (let i = 0; i < maturePeers.length; i += 1) {
    const tr = maturePeers.filter((_, j) => j !== i);
    const m = fitSurvivalTokenOdds(tr, tauMedDays);
    const pHat = m.predictLifetime(maturePeers[i].cat);
    brierLoo += Math.pow(maturePeers[i].tok - pHat, 2) / maturePeers.length;
  }
  const oddsBrierSkillPct = Number(((1 - brierLoo / Math.max(1e-6, brierBase)) * 100).toFixed(1));

  const scoredRows = targets.map(({ p, tier }) => {
    const cat = p.category ?? 'Uncategorized';
    const ageDays = daysBetween(p.listed_at, asOfMs);
    const featInput = {
      xTvl: Math.log10(Math.max(1, p.tvl_usd)),
      xChg: clampMomentum(p.change_7d),
      xAge: ageDays !== null ? Math.log10(ageDays + 30) : meanLogAge,
      xChains: Math.log10(Math.max(1, p.chains.length)),
    };
    const vec = winner.spec.features(featInput);
    const catShift = finalCatAdj ? (finalCatAdj.get(cat) ?? 0) : 0;
    const logPred = vec.reduce((s, v, idx) => s + v * finalWeights[idx], 0) + catShift;
    const impliedValue = Math.max(100_000, Math.round(Math.pow(10, logPred)));

    const w0 = weightMap.intercept ?? 0;
    const wTvl = weightMap['log10(TVL)'] ?? 0;
    const wChg = weightMap.chg_7d ?? 0;
    const wChains = weightMap['log10(chains)'] ?? 0;
    const wAge = weightMap['log10(age+30)'] ?? 0;
    const baseUsd = Math.round(Math.pow(10, w0 + wTvl * featInput.xTvl));
    const momentumMult = Number(Math.pow(10, wChg * featInput.xChg).toFixed(3));
    const chainsMult = Number(Math.pow(10, wChains * featInput.xChains).toFixed(3));
    const ageMult = Number(Math.pow(10, wAge * featInput.xAge).toFixed(3));
    const catMult = Number(Math.pow(10, catShift).toFixed(3));

    const disp = catDispScale.get(cat) ?? 1.0;
    const rowLow = Math.max(50_000, Math.round(impliedValue * Math.pow(10, q10Log * disp)));
    const rowHigh = Math.round(impliedValue * Math.pow(10, q90Log * disp));

    const st = catStats.get(cat) ?? { total: 0, tokened: 0 };
    const fittedOdds = fullOddsModel.predictConditional(cat, ageDays);
    const launchOdds = Number(Math.max(0.05, Math.min(0.95, fittedOdds)).toFixed(3));

    const expectedLaunchValue = Math.round(impliedValue * launchOdds);
    const { farm, steps } = buildFarmPlan(p, impliedValue, launchOdds, ageDays);

    const catComps = allPeerComps
      .filter((c) => c.category === cat)
      .sort((a, b) => Math.abs(Math.log10(a.tvl_usd) - featInput.xTvl) - Math.abs(Math.log10(b.tvl_usd) - featInput.xTvl));
    const fallbackComps = allPeerComps
      .filter((c) => c.category !== cat)
      .sort((a, b) => Math.abs(Math.log10(a.tvl_usd) - featInput.xTvl) - Math.abs(Math.log10(b.tvl_usd) - featInput.xTvl));

    return {
      slug: p.slug,
      name: p.name,
      rank: 0,
      implied_value_usd: impliedValue,
      farm_steps: steps,
      tier,
      category: cat,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      listed_at: p.listed_at,
      age_days: ageDays,
      conformal_low_usd: rowLow,
      conformal_high_usd: rowHigh,
      implied_multiple: Number((impliedValue / Math.max(1, p.tvl_usd)).toFixed(2)),
      peer_token_odds: launchOdds,
      peer_tokened_count: st.tokened,
      peer_total_count: st.total,
      expected_launch_value_usd: expectedLaunchValue,
      factor_breakdown: { base_usd: baseUsd, momentum_mult: momentumMult, chains_mult: chainsMult, age_mult: ageMult, cat_mult: catMult },
      farm,
      comps: [...catComps, ...fallbackComps].slice(0, 4),
    };
  });

  scoredRows.sort((a, b) => b.implied_value_usd - a.implied_value_usd || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1));
  scoredRows.forEach((row, idx) => { row.rank = idx + 1; });

  const formulaParts = winner.spec.terms.map((t, idx) => {
    if (idx >= finalWeights.length) return `+ ${t}`;
    const val = finalWeights[idx];
    if (idx === 0) return val.toFixed(3);
    return `${val >= 0 ? '+ ' : '- '}${Math.abs(val).toFixed(3)}·${t}`;
  });

  const ablation = ablationResults.map((a) => ({
    id: a.spec.id,
    label: a.spec.label,
    terms: a.spec.terms,
    loo_mdme: Number(a.loo.toFixed(2)),
    wf_mdme: Number(a.wf.toFixed(2)),
    cat_mdme: Number(a.catCv.toFixed(2)),
    geo_mdme: Number(a.geo.toFixed(2)),
    selected: a.spec.id === winner.spec.id,
  }));

  const radar = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    counts: {
      total_protocols: snapshot.protocols.length,
      watchlist: targets.filter((t) => t.tier === 'watchlist').length,
      early_discovery: targets.filter((t) => t.tier === 'early').length,
      scored_pretoken: scoredRows.length,
      live_tokened: liveTokenCount,
      excluded_wrappers: excludedWrappers,
      calibration_issuers: calib.length,
      dated_calibration_issuers: dated.length,
      calibration_categories: distinctCalibCats.length,
    },
    model: {
      selected_id: winner.spec.id,
      selected_label: winner.spec.label,
      formula_text: `log10(Implied USD) = ${formulaParts.join(' ')}`,
      weights: weightMap,
      loo_mdme: Number(winner.loo.toFixed(2)),
      wf_mdme: Number(winner.wf.toFixed(2)),
      cat_mdme: Number(winner.catCv.toFixed(2)),
      conformal_p10_mult: Number(confLowMult.toFixed(3)),
      conformal_p90_mult: Number(confHighMult.toFixed(3)),
      loo_cov80_pct: looCov80Pct,
      wf_cov80_pct: wfCov80Pct,
      cat_cov80_pct: catCov80Pct,
      odds_brier_loo: Number(brierLoo.toFixed(4)),
      odds_brier_base: Number(brierBase.toFixed(4)),
      odds_brier_skill_pct: oddsBrierSkillPct,
      ablation,
    },
    rows: scoredRows,
    live_token_sample: allPeerComps.slice(0, 12),
  };

  return { board: scoredRows, radar };
}

export function boardIssue(board, snapshot) {
  if (board.length < 100 || board.length > 4000) return `row count ${board.length} outside 100..4000`;
  const validSlugs = new Set(snapshot.protocols.map((p) => p.slug));
  const seen = new Set();
  let prevVal = Infinity;
  let allSame = true;
  const firstVal = board[0]?.implied_value_usd ?? 0;

  for (let i = 0; i < board.length; i += 1) {
    const r = board[i];
    if (!validSlugs.has(r.slug)) return `unknown slug ${r.slug}`;
    if (seen.has(r.slug)) return `duplicate slug ${r.slug}`;
    seen.add(r.slug);
    if (r.rank !== i + 1) return `rank mismatch at ${r.slug}`;
    if (!Number.isInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `bad value on ${r.slug}`;
    if (r.implied_value_usd > prevVal) return `non-monotonic value at ${r.slug}`;
    if (r.implied_value_usd !== firstVal) allSame = false;
    prevVal = r.implied_value_usd;
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `bad farm_steps count on ${r.slug}`;
    for (const s of r.farm_steps) {
      const len = s.trim().length;
      if (len < 20 || len > 600) return `farm_step length ${len} out of bounds on ${r.slug}`;
    }
  }
  if (allSame) return 'all implied values identical';
  for (const w of snapshot.watchlist) {
    if (!seen.has(w)) return `missing watchlist slug ${w}`;
  }
  return null;
}

export function dataScript(radar) {
  return `window.__RADAR_DATA__ = ${JSON.stringify(radar)};\n`;
}
