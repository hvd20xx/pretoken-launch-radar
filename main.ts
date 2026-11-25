// Pre-Token Launch Radar entrypoint (Node 22, standard library only).
// Reads /input/snapshot.json (or --input), runs dual-CV econometric valuation &
// costed farm/security modeling, writes board.json, data.js, dashboard/,
// screenshots/, and /output/dossier.json via vendor/makefun-board.ts.

import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './dashboard/model.mjs';
import { buildDossier } from './vendor/makefun-board.ts';

const ROOT_DIR = fileURLToPath(new URL('.', import.meta.url));
const DOSSIER_TEXT_MAX = 65_536;
const SHOT_CAPS = { max: 4, max_bytes: 524_288 };
const VALID_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

function readFlag(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && idx + 1 < process.argv.length) return process.argv[idx + 1]!;
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq !== undefined ? eq.slice(name.length + 3) : fallback;
}

function setModeSafe(p: string, mode: number): void {
  try {
    chmodSync(p, mode);
  } catch {
    // Non-POSIX filesystem
  }
}

function copyPlainDir(srcDir: string, dstDir: string): string[] {
  if (!existsSync(srcDir)) return [];
  mkdirSync(dstDir, { recursive: true, mode: 0o755 });
  setModeSafe(dstDir, 0o755);
  const files: string[] = [];
  for (const entry of readdirSync(srcDir).sort()) {
    if (entry.startsWith('.')) continue;
    const srcPath = join(srcDir, entry);
    if (!statSync(srcPath).isFile()) continue;
    const dstPath = join(dstDir, entry);
    copyFileSync(srcPath, dstPath);
    setModeSafe(dstPath, 0o644);
    files.push(entry);
  }
  return files;
}

function main(): void {
  const inputFlag = readFlag('input', '/input');
  const snapshotPath = inputFlag.endsWith('.json') ? inputFlag : join(inputFlag, 'snapshot.json');
  if (!existsSync(snapshotPath)) {
    process.stderr.write(`pretoken radar: snapshot not found at ${snapshotPath}\n`);
    process.exit(2);
  }

  const outDir = readFlag('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');
  const snapshot = parseSnapshot(readFileSync(snapshotPath, 'utf8'));
  const { board, radar } = buildRadar(snapshot);
  const problem = boardIssue(board, snapshot);
  if (problem !== null) {
    process.stderr.write(`pretoken radar: board validation failed: ${problem}\n`);
    process.exit(1);
  }

  mkdirSync(outDir, { recursive: true, mode: 0o755 });
  setModeSafe(outDir, 0o755);

  const boardOut = join(outDir, 'board.json');
  writeFileSync(boardOut, `${JSON.stringify({ rows: board })}\n`, 'utf8');
  setModeSafe(boardOut, 0o644);

  const dataOut = join(outDir, 'data.js');
  writeFileSync(dataOut, dataScript(radar), 'utf8');
  setModeSafe(dataOut, 0o644);

  const dashFiles = copyPlainDir(join(ROOT_DIR, 'dashboard'), join(outDir, 'dashboard'));
  const shotFiles = copyPlainDir(join(ROOT_DIR, 'screenshots'), join(outDir, 'screenshots'));

  if (shotFiles.length < 3) {
    process.stderr.write(`pretoken radar: expected 3-4 screenshots, found ${shotFiles.length}\n`);
    process.exit(1);
  }
  for (const name of shotFiles) {
    if (!VALID_NAME.test(name)) {
      process.stderr.write(`pretoken radar: invalid screenshot name ${name}\n`);
      process.exit(1);
    }
  }

  const dossierOut = join(outDir, 'dossier.json');
  writeFileSync(
    dossierOut,
    buildDossier(outDir, {
      text_max_bytes: DOSSIER_TEXT_MAX,
      screenshots: SHOT_CAPS,
      dashboard: { max_files: 400, max_bytes: 4_194_304 },
    }),
    'utf8',
  );
  setModeSafe(dossierOut, 0o644);

  process.stdout.write(
    `pretoken radar: scored ${board.length} protocols (${radar.counts.watchlist} watchlist + ${radar.counts.early_discovery} early; ` +
      `${radar.counts.live_tokened} live-tokened, ${radar.counts.excluded_wrappers} wrappers excluded)\n` +
      `model: ${radar.model.selected_label} | LOO x${radar.model.loo_mdme} | WF x${radar.model.wf_mdme} | Cat x${radar.model.cat_mdme} | ` +
      `wrote ${dashFiles.length} dashboard files, ${shotFiles.length} screenshots, dossier.json\n`,
  );
}

main();
