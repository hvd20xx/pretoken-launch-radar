// Empirical pre-token launch valuation, dual-CV ablation, conformal ranges,
// empirical peer launch odds, and $/hr costed farm + security audit engine.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface AblationRow {
  id: string;
  label: string;
  terms: string[];
  loo_mdme: number;
  wf_mdme: number;
  geo_mdme: number;
  selected: boolean;
}

export interface PeerComp {
  gecko_id: string;
  name: string;
  symbol: string;
  category: string;
  tvl_usd: number;
  mcap_usd: number;
  ratio: number;
}

export interface FarmEconomics {
  verdict: 'FARM' | 'MONITOR' | 'PASS';
  route_chain: string;
  min_capital_usd: number;
  est_gas_usd: number;
  est_hours: number;
  expected_ev_usd: number;
  ev_per_hour_usd: number;
  security_tier: 'LOW' | 'MODERATE' | 'ELEVATED';
  security_flags: string[];
}

export interface BoardRow {
  slug: string;
  name: string;
  rank: number;
  implied_value_usd: number;
  farm_steps: string[];
  // Extended analytical metadata for the dashboard
  tier: 'watchlist' | 'early';
  category: string;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  conformal_low_usd: number;
  conformal_high_usd: number;
  implied_multiple: number;
  peer_token_odds: number;
  peer_tokened_count: number;
  peer_total_count: number;
  expected_launch_value_usd: number;
  factor_breakdown: {
    base_usd: number;
    momentum_mult: number;
    chains_mult: number;
    age_mult: number;
  };
  farm: FarmEconomics;
  comps: PeerComp[];
}

export interface RadarPayload {
  as_of: string;
  source: string;
  floor_usd: number;
  counts: {
    total_protocols: number;
    watchlist: number;
    early_discovery: number;
    scored_pretoken: number;
    live_tokened: number;
    excluded_wrappers: number;
    calibration_issuers: number;
    dated_calibration_issuers: number;
  };
  model: {
    selected_id: string;
    selected_label: string;
    formula_text: string;
    weights: Record<string, number>;
    loo_mdme: number;
    wf_mdme: number;
    conformal_p10_mult: number;
    conformal_p90_mult: number;
    ablation: AblationRow[];
  };
  rows: BoardRow[];
  live_token_sample: PeerComp[];
}

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

const CHAIN_GAS_USD: Record<string, number> = {
  Solana: 0.08,
  Base: 0.35,
  Arbitrum: 0.45,
  Optimism: 0.35,
  Polygon: 0.25,
  Sui: 0.12,
  Aptos: 0.12,
  Avalanche: 0.65,
  BSC: 0.40,
  Binance: 0.40,
  Linea: 0.55,
  Scroll: 0.55,
  Sonic: 0.15,
  Hyperliquid: 0.20,
  'Hyperliquid L1': 0.20,
  Starknet: 0.40,
  Monad: 0.25,
  Unichain: 0.30,
  Berachain: 0.35,
  Gnosis: 0.15,
  xDai: 0.15,
  Mantle: 0.30,
  Blast: 0.45,
  Mode: 0.35,
  Bitcoin: 12.0,
  Ethereum: 18.0,
};

export function parseSnapshot(raw: string): Snapshot {
  const data = JSON.parse(raw) as Snapshot;
  if (!data || !Array.isArray(data.protocols) || !Array.isArray(data.watchlist)) {
    throw new Error('Invalid snapshot.json structure');
  }
  return data;
}

function daysBetween(isoDate: string | null, asOfMs: number): number | null {
  if (!isoDate) return null;
  const ms = Date.parse(`${isoDate}T00:00:00Z`);
  if (!Number.isFinite(ms)) return null;
  return Math.max(1, Math.round((asOfMs - ms) / 86_400_000));
}

function clampMomentum(chg: number | null): number {
  if (chg === null || !Number.isFinite(chg)) return 0;
  return Math.max(-0.5, Math.min(1.0, chg / 100));
}

function isWrapperOrExcluded(p: SnapshotProtocol): boolean {
  const cat = p.category ?? '';
  if (EXCLUDED_CATEGORIES.has(cat)) return true;
  if (WRAPPED_WORD.test(p.name)) return true;
  if (EXCHANGE_PREFIX.test(p.name)) return true;
  if (VERSION_SUFFIX.test(p.name)) return true;
  return false;
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const m = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < m; c += 1) {
    let piv = c;
    for (let r = c + 1; r < m; r += 1) {
      if (Math.abs(M[r]![c]!) > Math.abs(M[piv]![c]!)) piv = r;
    }
    const tmp = M[c]!;
    M[c] = M[piv]!;
    M[piv] = tmp;
    const diag = M[c]![c]!;
    if (Math.abs(diag) < 1e-12) continue;
    for (let r = c + 1; r < m; r += 1) {
      const factor = M[r]![c]! / diag;
      for (let k = c; k <= m; k += 1) M[r]![k] = M[r]![k]! - factor * M[c]![k]!;
    }
  }
  const x = new Array<number>(m).fill(0);
  for (let r = m - 1; r >= 0; r -= 1) {
    let s = M[r]![m]!;
    for (let k = r + 1; k < m; k += 1) s -= M[r]![k]! * x[k]!;
    x[r] = Math.abs(M[r]![r]!) < 1e-12 ? 0 : s / M[r]![r]!;
  }
  return x;
}

/** Deterministic IRLS Huber regression with mild L2 shrinkage on slopes. */
function fitHuber(X: number[][], y: number[], ridge = 0.1, delta = 0.6): number[] {
  const n = X.length;
  const m = X[0]!.length;
  let wts = new Array<number>(n).fill(1);
  let w = new Array<number>(m).fill(0);

  for (let iter = 0; iter < 16; iter += 1) {
    const A = Array.from({ length: m }, () => new Array<number>(m).fill(0));
    const b = new Array<number>(m).fill(0);
    for (let i = 0; i < n; i += 1) {
      const wi = wts[i]!;
      const xi = X[i]!;
      const yi = y[i]!;
      for (let j = 0; j < m; j += 1) {
        b[j] = b[j]! + wi * xi[j]! * yi;
        for (let k = 0; k < m; k += 1) {
          A[j]![k] = A[j]![k]! + wi * xi[j]! * xi[k]!;
        }
      }
    }
    for (let j = 1; j < m; j += 1) A[j]![j] = A[j]![j]! + ridge;
    w = solveLinearSystem(A, b);
    wts = y.map((yi, i) => {
      let pred = 0;
      for (let j = 0; j < m; j += 1) pred += X[i]![j]! * w[j]!;
      const err = Math.abs(yi - pred);
      return err <= delta ? 1 : delta / err;
    });
  }
  return w;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
  return sorted[idx]!;
}

interface CalibRow {
  gecko_id: string;
  name: string;
  symbol: string;
  cat: string;
  tvl_usd: number;
  mcap_usd: number;
  y: number;
  xTvl: number;
  xChg: number;
  xAge: number;
  xChains: number;
  listedMs: number | null;
}

interface CandidateSpec {
  id: string;
  label: string;
  terms: string[];
  features: (r: { xTvl: number; xChg: number; xAge: number; xChains: number }) => number[];
}

const CANDIDATE_SPECS: CandidateSpec[] = [
  {
    id: 'tvl_only',
    label: 'TVL power-law baseline',
    terms: ['intercept', 'log10(TVL)'],
    features: (r) => [1, r.xTvl],
  },
  {
    id: 'tvl_mom',
    label: 'TVL + 7d momentum',
    terms: ['intercept', 'log10(TVL)', 'chg_7d'],
    features: (r) => [1, r.xTvl, r.xChg],
  },
  {
    id: 'tvl_chains',
    label: 'TVL + chain breadth',
    terms: ['intercept', 'log10(TVL)', 'log10(chains)'],
    features: (r) => [1, r.xTvl, r.xChains],
  },
  {
    id: 'tvl_age',
    label: 'TVL + tenure',
    terms: ['intercept', 'log10(TVL)', 'log10(age+30)'],
    features: (r) => [1, r.xTvl, r.xAge],
  },
  {
    id: 'tvl_mom_chains',
    label: 'TVL + 7d momentum + chain breadth',
    terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(chains)'],
    features: (r) => [1, r.xTvl, r.xChg, r.xChains],
  },
  {
    id: 'tvl_mom_age_chains',
    label: 'Full 4-factor (TVL + momentum + tenure + chains)',
    terms: ['intercept', 'log10(TVL)', 'chg_7d', 'log10(age+30)', 'log10(chains)'],
    features: (r) => [1, r.xTvl, r.xChg, r.xAge, r.xChains],
  },
];

function pickCheapestChain(chains: string[]): { chain: string; gas: number } {
  if (chains.length === 0) return { chain: 'Ethereum', gas: 18.0 };
  let bestChain = chains[0]!;
  let bestGas = CHAIN_GAS_USD[bestChain] ?? 2.5;
  for (const c of chains) {
    const g = CHAIN_GAS_USD[c] ?? 2.5;
    if (g < bestGas) {
      bestGas = g;
      bestChain = c;
    }
  }
  return { chain: bestChain, gas: bestGas };
}

function fmtUsdShort(n: number): string {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

function buildFarmPlan(
  p: SnapshotProtocol,
  impliedValue: number,
  launchOdds: number,
  ageDays: number | null,
): { farm: FarmEconomics; steps: string[] } {
  const cat = p.category ?? 'DeFi';
  const { chain, gas } = pickCheapestChain(p.chains);
  const isSolana = p.chains.some((c) => c.toLowerCase() === 'solana');
  const isMainnetOnly = p.chains.length === 1 && p.chains[0] === 'Ethereum';

  // Size minimum viable capital so chain gas stays <2-3% of stake and clears Sybil dust floors
  const minCapital = isMainnetOnly
    ? p.tvl_usd >= 200_000_000
      ? 5000
      : 2500
    : p.tvl_usd >= 200_000_000
      ? 2000
      : p.tvl_usd >= 50_000_000
        ? 1000
        : 500;

  // 12-week active farming cadence: initial security audit/deposit + 6 biweekly checks + claim
  const estHours = cat === 'Dexs' || cat === 'Derivatives' || cat === 'Perps' ? 4.0 : isMainnetOnly ? 2.5 : 3.0;
  const totalGas = Number((gas * (estHours >= 4 ? 8 : 4)).toFixed(2));

  // Two-component transparent airdrop capture (7.5% community pool: 60% TVL pro-rata + 40% active-wallet tier):
  // Active wallet pool scales sub-linearly with TVL (estWallets = max(800, sqrt(tvl_usd) * 1.4)),
  // discounted by empirical peer token-launch odds and conservatively capped at 35% of deployed capital.
  const proRataEv = (minCapital / Math.max(p.tvl_usd, 5_000_000)) * (impliedValue * 0.045);
  const estActiveWallets = Math.max(800, Math.round(Math.sqrt(Math.max(p.tvl_usd, 5_000_000)) * 1.4));
  const tierEv = (impliedValue * 0.03) / estActiveWallets;
  const grossEv = Math.min((proRataEv + tierEv) * launchOdds, minCapital * 0.35);

  // Deduct 90-day opportunity cost (4% annualized => 1.0% over 90d) + execution gas
  const oppCost = minCapital * 0.01;
  const netEv = Math.round((grossEv - totalGas - oppCost) * 100) / 100;
  const evPerHour = Math.round((netEv / estHours) * 100) / 100;

  // Pre-deposit security audit flags (informed by smart-contract & Solana SPL audit rules)
  const flags: string[] = [];
  if (p.change_7d !== null && p.change_7d <= -15) {
    flags.push(`Sharp 7d TVL outflow (${p.change_7d.toFixed(1)}%) — verify withdrawal liquidity before deposit`);
  }
  if (ageDays !== null && ageDays < 90) {
    flags.push(`Young protocol (${ageDays}d tracked) — cap exposure until first independent audit settles`);
  }
  if (cat === 'Risk Curators' || cat === 'Onchain Capital Allocator' || cat === 'Yield') {
    flags.push('Curator/vault admin role risk — inspect timelock delay and strategy whitelist permissions');
  }
  if (isSolana) {
    flags.push('Solana SPL check — verify program upgrade authority, freeze/mint flags & Token-2022 hooks');
  } else {
    flags.push('EVM contract check — confirm verified explorer source, proxy admin multisig & withdraw path');
  }

  const securityTier: 'LOW' | 'MODERATE' | 'ELEVATED' =
    (p.change_7d !== null && p.change_7d <= -20) || (ageDays !== null && ageDays < 60)
      ? 'ELEVATED'
      : flags.length >= 2
        ? 'MODERATE'
        : 'LOW';

  let verdict: 'FARM' | 'MONITOR' | 'PASS';
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

  const cadenceStep = `Execution budget: route via ${chain} (~$${totalGas.toFixed(2)} est. total gas across 12 weeks, ~${estHours.toFixed(1)}h total time) — expected net EV ${netEv >= 0 ? '+' : '-'}$${Math.abs(netEv).toFixed(2)} (${evPerHour >= 0 ? '+' : '-'}$${Math.abs(evPerHour).toFixed(2)}/hr, verdict: ${verdict})`;

  const monitorStep =
    p.change_7d !== null && p.change_7d < -10
      ? `Risk trigger: ${p.name} saw ${p.change_7d.toFixed(1)}% 7d TVL drawdown; pause top-ups until 7d TVL stabilizes and exit immediately if TVL drops another 25%`
      : `Snapshot discipline: track ${p.name}'s ${fmtUsdShort(p.tvl_usd)} TVL and peer token odds (${Math.round(launchOdds * 100)}% in ${cat}); rebalance biweekly and exit if TGE terms dilute small wallets`;

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
      security_tier: securityTier,
      security_flags: flags,
    },
    steps: [secStep, catAction, cadenceStep, monitorStep, claimStep],
  };
}

export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarPayload } {
  const asOfMs = Date.parse(`${snapshot.as_of}T00:00:00Z`);
  const watchlistSet = new Set(snapshot.watchlist);

  // 1. Category empirical token-issuance odds across all 758 protocols (Laplace-smoothed toward global rate)
  const catStats = new Map<string, { total: number; tokened: number }>();
  let globalTotal = 0;
  let globalTokened = 0;
  let liveTokenCount = 0;

  for (const p of snapshot.protocols) {
    const hasToken = p.gecko_id !== null || p.symbol !== null;
    if (hasToken) liveTokenCount += 1;
    const cat = p.category ?? 'Uncategorized';
    const st = catStats.get(cat) ?? { total: 0, tokened: 0 };
    st.total += 1;
    if (hasToken) st.tokened += 1;
    catStats.set(cat, st);
    globalTotal += 1;
    if (hasToken) globalTokened += 1;
  }
  const globalTokenRate = globalTotal > 0 ? globalTokened / globalTotal : 0.5;

  // 2. Build deduplicated tokened calibration cohort (aggregate ALL sub-versions' TVL by gecko_id)
  const byGecko = new Map<
    string,
    {
      gecko_id: string;
      name: string;
      symbol: string;
      mcap_usd: number;
      tvl_usd: number;
      wChg: number;
      chgW: number;
      minListedMs: number | null;
      chains: Set<string>;
      cat: string;
      allExcl: boolean;
      maxPartTvl: number;
    }
  >();

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
      if (listedMs !== null && (prev.minListedMs === null || listedMs < prev.minListedMs)) {
        prev.minListedMs = listedMs;
      }
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
    .map((g) => Math.log10(Math.max(1, (asOfMs - g.minListedMs!) / 86_400_000) + 30));
  const meanLogAge = datedAges.length > 0 ? datedAges.reduce((a, b) => a + b, 0) / datedAges.length : 2.5;

  const calib: CalibRow[] = rawCalib
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
      xAge:
        g.minListedMs !== null
          ? Math.log10(Math.max(1, (asOfMs - g.minListedMs) / 86_400_000) + 30)
          : meanLogAge,
      xChains: Math.log10(Math.max(1, g.chains.size)),
      listedMs: g.minListedMs,
    }))
    .sort((a, b) => (a.gecko_id < b.gecko_id ? -1 : 1));

  // 3. Dual held-out validation (Leave-One-Out CV + Temporal Walk-Forward CV)
  const dated = calib.filter((d) => d.listedMs !== null).sort((a, b) => a.listedMs! - b.listedMs!);
  const undated = calib.filter((d) => d.listedMs === null);

  const ablationResults: Array<{ spec: CandidateSpec; loo: number; wf: number; geo: number; looResiduals: number[] }> =
    CANDIDATE_SPECS.map((spec) => {
      const looResiduals: number[] = [];
      for (let i = 0; i < calib.length; i += 1) {
        const train = calib.filter((_, j) => j !== i);
        const w = fitHuber(
          train.map((r) => spec.features(r)),
          train.map((r) => r.y),
        );
        const xi = spec.features(calib[i]!);
        const pred = xi.reduce((s, v, idx) => s + v * w[idx]!, 0);
        looResiduals.push(calib[i]!.y - pred);
      }
      const absLoo = looResiduals.map((r) => Math.abs(r)).sort((a, b) => a - b);
      const looMdme = Math.pow(10, quantile(absLoo, 0.5));

      const wfAbs: number[] = [];
      const minTrain = Math.min(20, Math.floor(dated.length / 2));
      for (let i = minTrain; i < dated.length; i += 1) {
        const train = [...undated, ...dated.slice(0, i)];
        const w = fitHuber(
          train.map((r) => spec.features(r)),
          train.map((r) => r.y),
        );
        const xi = spec.features(dated[i]!);
        const pred = xi.reduce((s, v, idx) => s + v * w[idx]!, 0);
        wfAbs.push(Math.abs(dated[i]!.y - pred));
      }
      wfAbs.sort((a, b) => a - b);
      const wfMdme = Math.pow(10, quantile(wfAbs, 0.5));
      return { spec, loo: looMdme, wf: wfMdme, geo: Math.sqrt(looMdme * wfMdme), looResiduals };
    });

  // Pick the model with lowest geometric-mean out-of-sample multiplicative error
  ablationResults.sort((a, b) => a.geo - b.geo);
  const winner = ablationResults[0]!;
  const finalWeights = fitHuber(
    calib.map((r) => winner.spec.features(r)),
    calib.map((r) => r.y),
  );
  const weightMap: Record<string, number> = {};
  winner.spec.terms.forEach((t, idx) => {
    weightMap[t] = Number(finalWeights[idx]!.toFixed(4));
  });

  // Conformal 80% prediction multipliers directly from signed out-of-sample LOO residuals
  const sortedSignedRes = [...winner.looResiduals].sort((a, b) => a - b);
  const confLowMult = Math.pow(10, quantile(sortedSignedRes, 0.1));
  const confHighMult = Math.pow(10, quantile(sortedSignedRes, 0.9));

  // Build peer comps lookup by category (and global fallback)
  const allPeerComps: PeerComp[] = calib
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

  // 4. Separate clean pre-token cohort (Watchlist 40 + Early Discovery below floor) vs Excluded Wrappers
  let excludedWrappers = 0;
  const targets: Array<{ p: SnapshotProtocol; tier: 'watchlist' | 'early' }> = [];

  for (const p of snapshot.protocols) {
    if (watchlistSet.has(p.slug)) {
      targets.push({ p, tier: 'watchlist' });
    } else if (p.gecko_id === null && p.symbol === null) {
      if (isWrapperOrExcluded(p)) {
        excludedWrappers += 1;
      } else {
        targets.push({ p, tier: 'early' });
      }
    }
  }

  // 5. Score every pre-token target with the winning dual-CV model
  const scoredRows: BoardRow[] = targets.map(({ p, tier }) => {
    const cat = p.category ?? 'Uncategorized';
    const ageDays = daysBetween(p.listed_at, asOfMs);
    const featInput = {
      xTvl: Math.log10(Math.max(1, p.tvl_usd)),
      xChg: clampMomentum(p.change_7d),
      xAge: ageDays !== null ? Math.log10(ageDays + 30) : meanLogAge,
      xChains: Math.log10(Math.max(1, p.chains.length)),
    };
    const vec = winner.spec.features(featInput);
    const logPred = vec.reduce((s, v, idx) => s + v * finalWeights[idx]!, 0);
    const impliedValue = Math.max(100_000, Math.round(Math.pow(10, logPred)));

    // Decompose multiplicative contributions transparently
    const w0 = weightMap.intercept ?? 0;
    const wTvl = weightMap['log10(TVL)'] ?? 0;
    const wChg = weightMap.chg_7d ?? 0;
    const wChains = weightMap['log10(chains)'] ?? 0;
    const wAge = weightMap['log10(age+30)'] ?? 0;
    const baseUsd = Math.round(Math.pow(10, w0 + wTvl * featInput.xTvl));
    const momentumMult = Number(Math.pow(10, wChg * featInput.xChg).toFixed(3));
    const chainsMult = Number(Math.pow(10, wChains * featInput.xChains).toFixed(3));
    const ageMult = Number(Math.pow(10, wAge * featInput.xAge).toFixed(3));

    // Empirical category peer token-issuance odds (shrunk with pseudo-count m=4)
    const st = catStats.get(cat) ?? { total: 0, tokened: 0 };
    const rawOdds = (st.tokened + 4 * globalTokenRate) / (st.total + 4);
    // Gentle tenure calibration: protocols running >3 years without a token have slightly lower marginal launch hazard
    const tenureFactor = ageDays !== null && ageDays > 1095 ? 0.85 : ageDays !== null && ageDays < 180 ? 1.05 : 1.0;
    const launchOdds = Number(Math.max(0.05, Math.min(0.95, rawOdds * tenureFactor)).toFixed(3));

    const expectedLaunchValue = Math.round(impliedValue * launchOdds);
    const { farm, steps } = buildFarmPlan(p, impliedValue, launchOdds, ageDays);

    // Closest 4 live-token peers in the same category (or closest TVL overall if <4)
    const catComps = allPeerComps
      .filter((c) => c.category === cat)
      .sort((a, b) => Math.abs(Math.log10(a.tvl_usd) - featInput.xTvl) - Math.abs(Math.log10(b.tvl_usd) - featInput.xTvl));
    const fallbackComps = allPeerComps
      .filter((c) => c.category !== cat)
      .sort((a, b) => Math.abs(Math.log10(a.tvl_usd) - featInput.xTvl) - Math.abs(Math.log10(b.tvl_usd) - featInput.xTvl));
    const comps = [...catComps, ...fallbackComps].slice(0, 4);

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
      conformal_low_usd: Math.max(50_000, Math.round(impliedValue * confLowMult)),
      conformal_high_usd: Math.round(impliedValue * confHighMult),
      implied_multiple: Number((impliedValue / Math.max(1, p.tvl_usd)).toFixed(2)),
      peer_token_odds: launchOdds,
      peer_tokened_count: st.tokened,
      peer_total_count: st.total,
      expected_launch_value_usd: expectedLaunchValue,
      factor_breakdown: {
        base_usd: baseUsd,
        momentum_mult: momentumMult,
        chains_mult: chainsMult,
        age_mult: ageMult,
      },
      farm,
      comps,
    };
  });

  // Sort descending by implied_value_usd (tie-break by tvl_usd desc, then slug asc)
  scoredRows.sort((a, b) => b.implied_value_usd - a.implied_value_usd || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1));
  scoredRows.forEach((row, idx) => {
    row.rank = idx + 1;
  });

  const formulaParts = winner.spec.terms.map((t, idx) => {
    const val = finalWeights[idx]!;
    if (idx === 0) return val.toFixed(3);
    return `${val >= 0 ? '+ ' : '- '}${Math.abs(val).toFixed(3)}·${t}`;
  });

  const ablation: AblationRow[] = ablationResults.map((a) => ({
    id: a.spec.id,
    label: a.spec.label,
    terms: a.spec.terms,
    loo_mdme: Number(a.loo.toFixed(2)),
    wf_mdme: Number(a.wf.toFixed(2)),
    geo_mdme: Number(a.geo.toFixed(2)),
    selected: a.spec.id === winner.spec.id,
  }));

  const radar: RadarPayload = {
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
    },
    model: {
      selected_id: winner.spec.id,
      selected_label: winner.spec.label,
      formula_text: `log10(Implied USD) = ${formulaParts.join(' ')}`,
      weights: weightMap,
      loo_mdme: Number(winner.loo.toFixed(2)),
      wf_mdme: Number(winner.wf.toFixed(2)),
      conformal_p10_mult: Number(confLowMult.toFixed(3)),
      conformal_p90_mult: Number(confHighMult.toFixed(3)),
      ablation,
    },
    rows: scoredRows,
    live_token_sample: allPeerComps.slice(0, 12),
  };

  return { board: scoredRows, radar };
}

export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  if (board.length < 100 || board.length > 4000) return `row count ${board.length} outside 100..4000`;
  const validSlugs = new Set(snapshot.protocols.map((p) => p.slug));
  const seen = new Set<string>();
  let prevVal = Infinity;
  let allSame = true;
  const firstVal = board[0]?.implied_value_usd ?? 0;

  for (let i = 0; i < board.length; i += 1) {
    const r = board[i]!;
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

export function dataScript(radar: RadarPayload): string {
  return `window.__RADAR_DATA__ = ${JSON.stringify(radar)};\n`;
}
