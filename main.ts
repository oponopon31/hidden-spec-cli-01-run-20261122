// mend: applies AI-written edits to a tree of text files.
//
// Written from SPEC.md alone. Implements the specification's sections 1-10:
// invocation, output and exit codes, files/lines/paths, format detection,
// three-tier line matching, unified diffs, SEARCH/REPLACE blocks, V4A
// patches, JSON edits, and order plus atomicity.
//
// Node 22, standard library only. Deterministic: no times, no randomness.

import { readFileSync, writeFileSync, mkdirSync, unlinkSync } from 'node:fs';
import { dirname } from 'node:path';

type FormatName = 'unified' | 'search-replace' | 'v4a' | 'json' | 'auto';

class ParseFault extends Error {
  line: number;
  constructor(line: number) {
    super('parse ' + line);
    this.line = line;
  }
}

interface EditFailure {
  code: string;
  path: string;
}

interface ApplyOutcome {
  failed: EditFailure | null;
  move: { old: string; next: string } | null;
}

function okOutcome(): ApplyOutcome {
  return { failed: null, move: null };
}

function failOutcome(code: string, path: string): ApplyOutcome {
  return { failed: { code, path }, move: null };
}

// ---- text helpers (SPEC section 3) ----

// A patch's lines: split at \n, one trailing \r removed from each, without
// the empty piece after a final \n.
function splitPatchLines(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n').map((p) => (p.endsWith('\r') ? p.slice(0, -1) : p));
  if (text.endsWith('\n')) parts.pop();
  return parts;
}

function isSafePath(p: string): boolean {
  if (p === '') return false;
  if (p.startsWith('/')) return false;
  if (p.includes('\\')) return false;
  const segs = p.split('/');
  for (const s of segs) {
    if (s === '' || s === '.' || s === '..') return false;
  }
  return true;
}

interface FileState {
  lines: string[];
  brk: string;
  ends: boolean;
}

// Split a file's text into its lines, its line break and whether it ends
// with a line break. The break is CRLF when the first line feed follows a
// carriage return, LF otherwise. An empty file has no lines.
function parseFileText(text: string): FileState {
  if (text === '') return { lines: [], brk: '\n', ends: false };
  const firstLf = text.indexOf('\n');
  const brk = firstLf > 0 && text[firstLf - 1] === '\r' ? '\r\n' : '\n';
  const ends = text.endsWith(brk);
  const body = ends ? text.slice(0, -brk.length) : text;
  return { lines: body.split(brk), brk, ends };
}

function renderFileText(lines: string[], brk: string, ends: boolean): string {
  if (lines.length === 0) return '';
  return lines.join(brk) + (ends ? brk : '');
}

// ---- three-tier line matching (SPEC section 5) ----

function trimEndWs(s: string): string {
  return s.replace(/[ \t]+$/, '');
}

function trimWs(s: string): string {
  return s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
}

function lineEq(a: string, b: string, tier: number): boolean {
  if (tier === 0) return a === b;
  if (tier === 1) return trimEndWs(a) === trimEndWs(b);
  return trimWs(a) === trimWs(b);
}

function findOccurrences(lines: string[], oldLines: string[], tier: number): number[] {
  const out: number[] = [];
  if (oldLines.length === 0 || oldLines.length > lines.length) return out;
  for (let i = 0; i + oldLines.length <= lines.length; i++) {
    let good = true;
    for (let j = 0; j < oldLines.length; j++) {
      if (!lineEq(lines[i + j], oldLines[j], tier)) {
        good = false;
        break;
      }
    }
    if (good) out.push(i);
  }
  return out;
}

// The first tier at which the old lines occur at least once, with its
// positions; null when they occur at no tier. --strict tries tier 0 only.
function findTier(lines: string[], oldLines: string[], strict: boolean): { tier: number; positions: number[] } | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    const p = findOccurrences(lines, oldLines, t);
    if (p.length > 0) return { tier: t, positions: p };
  }
  return null;
}

function leadingWs(s: string): string {
  const m = /^[ \t]*/.exec(s);
  return m ? m[0] : '';
}

interface NewLine {
  text: string;
  // Index into O of the context line this N line is, or null for a pure
  // addition. A context line is written as the file's own matched line.
  ctx: number | null;
}

// Build the lines that replace a match: context lines come from the file,
// other lines are written as they are, except at tier 2 where they are
// re-indented from the old lines' first indent to the file's.
function buildReplacement(
  fileLines: string[],
  pos: number,
  oldLines: string[],
  newLines: NewLine[],
  tier: number,
): string[] {
  const out: string[] = [];
  const f = leadingWs(fileLines[pos]);
  const p = leadingWs(oldLines[0]);
  for (const item of newLines) {
    if (item.ctx !== null) {
      out.push(fileLines[pos + item.ctx]);
      continue;
    }
    const t = item.text;
    if (tier === 2 && t !== '' && !/^[ \t]+$/.test(t) && t.startsWith(p)) {
      out.push(f + t.slice(p.length));
    } else {
      out.push(t);
    }
  }
  return out;
}

// ---- the tree under edit ----

// The tree as the patch leaves it, over the tree as it started. Files are
// read from disk lazily, so an edit sees what earlier edits left.
class Tree {
  private origCache = new Map<string, string | null>();
  private curMap = new Map<string, string | null>();
  private touchedSet = new Set<string>();

  origOf(p: string): string | null {
    if (!this.origCache.has(p)) {
      let v: string | null;
      try {
        v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.origCache.set(p, v);
    }
    const v = this.origCache.get(p);
    return v === undefined ? null : v;
  }

  get(p: string): string | null {
    if (this.curMap.has(p)) {
      const v = this.curMap.get(p);
      return v === undefined ? null : v;
    }
    return this.origOf(p);
  }

  exists(p: string): boolean {
    return this.get(p) !== null;
  }

  set(p: string, v: string | null): void {
    this.curMap.set(p, v);
    this.touchedSet.add(p);
  }

  touched(): string[] {
    return [...this.touchedSet];
  }
}

// ---- unified diffs (SPEC section 6) ----

interface Hunk {
  // The -A number of the hunk header.
  a: number;
  // O: the hunk's context and - lines, in body order.
  o: string[];
  // N: the hunk's context and + lines, in body order.
  n: NewLine[];
  // A \ line follows the hunk's last context or + line.
  noBreak: boolean;
  // A \ line follows the hunk's last + line.
  plusNoBreak: boolean;
}

function buildHunk(a: number, body: string[]): Hunk {
  const o: string[] = [];
  const n: NewLine[] = [];
  let lastCtxPlus = -1;
  let lastPlus = -1;
  for (let i = 0; i < body.length; i++) {
    const line = body[i];
    if (line === '') {
      o.push('');
      n.push({ text: '', ctx: o.length - 1 });
      lastCtxPlus = i;
    } else if (line[0] === ' ') {
      const t = line.slice(1);
      o.push(t);
      n.push({ text: t, ctx: o.length - 1 });
      lastCtxPlus = i;
    } else if (line[0] === '-') {
      o.push(line.slice(1));
    } else if (line[0] === '+') {
      n.push({ text: line.slice(1), ctx: null });
      lastCtxPlus = i;
      lastPlus = i;
    }
    // A \ line refers to the body line before it; it adds no line itself.
  }
  const follows = (idx: number): boolean =>
    idx >= 0 && idx + 1 < body.length && body[idx + 1][0] === '\\';
  return { a, o, n, noBreak: follows(lastCtxPlus), plusNoBreak: follows(lastPlus) };
}

interface SectionDraft {
  // null means /dev/null (no file); undefined means not named.
  oldPath: string | null | undefined;
  newPath: string | null | undefined;
  renameFrom: string | undefined;
  renameTo: string | undefined;
  diffOld: string | undefined;
  diffNew: string | undefined;
  newFile: boolean;
  deletedFile: boolean;
  hasPair: boolean;
  hunks: Hunk[];
}

function newSection(): SectionDraft {
  return {
    oldPath: undefined,
    newPath: undefined,
    renameFrom: undefined,
    renameTo: undefined,
    diffOld: undefined,
    diffNew: undefined,
    newFile: false,
    deletedFile: false,
    hasPair: false,
    hunks: [],
  };
}

function stripAB(p: string): string {
  if (p.startsWith('a/') || p.startsWith('b/')) return p.slice(2);
  return p;
}

// The path of a --- or +++ header line: text after the four characters, up
// to the first tab, with surrounding spaces removed; /dev/null is null.
function headerPath(line: string): string | null {
  let rest = line.slice(4);
  const tab = rest.indexOf('\t');
  if (tab >= 0) rest = rest.slice(0, tab);
  rest = rest.trim();
  if (rest === '/dev/null') return null;
  return stripAB(rest);
}

const DIFF_GIT_RE = /^diff --git a\/(.+) b\/(.+)$/;
const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$/;

function parseUnified(lines: string[]): EditDesc[] {
  const sections: SectionDraft[] = [];
  let cur: SectionDraft | null = null;
  const pushCurrent = (): void => {
    if (cur) sections.push(cur);
    cur = null;
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const lineNo = i + 1;
    if (line === 'diff --git' || line.startsWith('diff --git ')) {
      const m = DIFF_GIT_RE.exec(line);
      if (!m) throw new ParseFault(lineNo);
      pushCurrent();
      cur = newSection();
      cur.diffOld = m[1];
      cur.diffNew = m[2];
      i++;
      continue;
    }
    if (line.startsWith('@@')) {
      if (!cur) throw new ParseFault(lineNo);
      const m = HUNK_RE.exec(line);
      if (!m) throw new ParseFault(lineNo);
      const a = parseInt(m[1], 10);
      let j = i + 1;
      const body: string[] = [];
      while (j < lines.length) {
        const l = lines[j];
        // A --- line followed by a +++ line ends the body: it starts a
        // section. Any other - line is an old line of this hunk.
        if (l.startsWith('--- ') && j + 1 < lines.length && lines[j + 1].startsWith('+++ ')) break;
        if (l === '' || l[0] === ' ' || l[0] === '-' || l[0] === '+' || l[0] === '\\') {
          body.push(l);
          j++;
          continue;
        }
        break;
      }
      while (body.length > 0 && body[body.length - 1] === '') body.pop();
      cur.hunks.push(buildHunk(a, body));
      i = j;
      continue;
    }
    if (line.startsWith('--- ')) {
      const nextIsPlus = i + 1 < lines.length && lines[i + 1].startsWith('+++ ');
      if (nextIsPlus) {
        if (cur && cur.hunks.length === 0 && !cur.hasPair) {
          cur.oldPath = headerPath(line);
          cur.newPath = headerPath(lines[i + 1]);
          cur.hasPair = true;
          i += 2;
          continue;
        }
        pushCurrent();
        cur = newSection();
        cur.oldPath = headerPath(line);
        cur.newPath = headerPath(lines[i + 1]);
        cur.hasPair = true;
        i += 2;
        continue;
      }
      // A --- line not followed by a +++ line, inside a section's header.
      if (cur && cur.hunks.length === 0) throw new ParseFault(lineNo + 1);
      i++;
      continue;
    }
    if (cur && cur.hunks.length === 0) {
      if (line.startsWith('+++ ')) {
        if (!cur.hasPair && cur.newPath === undefined) cur.newPath = headerPath(line);
        i++;
        continue;
      }
      if (line.startsWith('rename from ')) {
        cur.renameFrom = line.slice(12).trim();
        i++;
        continue;
      }
      if (line.startsWith('rename to ')) {
        cur.renameTo = line.slice(10).trim();
        i++;
        continue;
      }
      if (line.startsWith('new file mode')) {
        cur.newFile = true;
        i++;
        continue;
      }
      if (line.startsWith('deleted file mode')) {
        cur.deletedFile = true;
        i++;
        continue;
      }
      // index, old mode, new mode, similarity index, dissimilarity index,
      // Index: lines and any other header line are ignored.
      i++;
      continue;
    }
    // Outside a section's header: prose and code fences are ignored.
    i++;
  }
  pushCurrent();
  if (sections.length === 0) throw new ParseFault(1);
  return sections.map(sectionToEdit);
}

type EditDesc =
  | { kind: 'u-create'; path: string; plusLines: string[]; noBreak: boolean }
  | { kind: 'u-delete'; path: string }
  | { kind: 'u-move'; oldPath: string; newPath: string; hunks: Hunk[] }
  | { kind: 'u-change'; path: string; hunks: Hunk[] }
  | { kind: 'u-noop' }
  | { kind: 'j-create'; path: string; text: string }
  | { kind: 'j-str-replace'; path: string; oldStr: string; newStr: string }
  | { kind: 'j-insert'; path: string; insertLine: number; newStr: string }
  | { kind: 's-replace'; path: string; oldLines: string[]; newLines: string[] }
  | { kind: 'v-add'; path: string; lines: string[] }
  | { kind: 'v-update'; path: string; moveTo: string | undefined; chunks: VChunk[] };

// One chunk of a V4A Update File section: O is its context and - lines, N
// its context and + lines (context lines carry their index into O).
interface VChunk {
  // null means no anchor.
  anchor: string | null;
  o: string[];
  n: NewLine[];
  // True when a *** End of File line closed this chunk.
  eof: boolean;
}

function sectionToEdit(sec: SectionDraft): EditDesc {
  const oldP =
    sec.oldPath !== undefined
      ? sec.oldPath
      : sec.renameFrom !== undefined
        ? sec.renameFrom
        : sec.diffOld;
  const newP =
    sec.newPath !== undefined
      ? sec.newPath
      : sec.renameTo !== undefined
        ? sec.renameTo
        : sec.diffNew;
  const isCreate = oldP === null || sec.newFile;
  const isDelete = newP === null || sec.deletedFile;
  if (isCreate) {
    if (newP === null || newP === undefined) return { kind: 'u-noop' };
    const plusLines: string[] = [];
    let noBreak = false;
    let sawPlus = false;
    for (const h of sec.hunks) {
      for (const item of h.n) {
        if (item.ctx === null) plusLines.push(item.text);
      }
      if (h.n.some((item) => item.ctx === null)) {
        sawPlus = true;
        noBreak = h.plusNoBreak;
      }
    }
    return { kind: 'u-create', path: newP, plusLines, noBreak: sawPlus ? noBreak : false };
  }
  if (isDelete) {
    if (oldP === null || oldP === undefined) return { kind: 'u-noop' };
    return { kind: 'u-delete', path: oldP };
  }
  if (oldP === undefined || newP === undefined) return { kind: 'u-noop' };
  if (oldP !== newP) return { kind: 'u-move', oldPath: oldP, newPath: newP, hunks: sec.hunks };
  return { kind: 'u-change', path: oldP, hunks: sec.hunks };
}

// Apply a section's hunks in order to a file's state, tracking d, the sum
// of (N length minus O length) over the hunks applied so far. Returns false
// when a hunk finds no match.
function applyHunksToState(fs: FileState, hunks: Hunk[], strict: boolean): boolean {
  let d = 0;
  for (const h of hunks) {
    if (h.o.length === 0) {
      const insertPos = Math.max(0, Math.min(h.a + d, fs.lines.length));
      const atEnd = insertPos === fs.lines.length;
      const newLines = h.n.map((item) => item.text);
      fs.lines.splice(insertPos, 0, ...newLines);
      if (atEnd) fs.ends = !h.noBreak;
      d += h.n.length;
      continue;
    }
    const found = findTier(fs.lines, h.o, strict);
    if (!found) return false;
    const target = h.a - 1 + d;
    let best = found.positions[0];
    let bestDist = Math.abs(best - target);
    for (const p of found.positions) {
      const dist = Math.abs(p - target);
      if (dist < bestDist) {
        best = p;
        bestDist = dist;
      }
    }
    const pos = best;
    const matchedToEnd = pos + h.o.length === fs.lines.length;
    const replacement = buildReplacement(fs.lines, pos, h.o, h.n, found.tier);
    fs.lines.splice(pos, h.o.length, ...replacement);
    if (matchedToEnd) fs.ends = !h.noBreak;
    d += h.n.length - h.o.length;
  }
  return true;
}

// ---- JSON edits (SPEC section 9) ----

function validateJsonEdit(obj: unknown, line: number): EditDesc {
  if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) throw new ParseFault(line);
  const rec = obj as Record<string, unknown>;
  if (typeof rec.path !== 'string') throw new ParseFault(line);
  if (typeof rec.command !== 'string') throw new ParseFault(line);
  const path = rec.path;
  if (rec.command === 'create') {
    if (typeof rec.file_text !== 'string') throw new ParseFault(line);
    return { kind: 'j-create', path, text: rec.file_text };
  }
  if (rec.command === 'str_replace') {
    if (typeof rec.old_str !== 'string' || rec.old_str.length === 0) throw new ParseFault(line);
    if (rec.new_str !== undefined && typeof rec.new_str !== 'string') throw new ParseFault(line);
    return {
      kind: 'j-str-replace',
      path,
      oldStr: rec.old_str,
      newStr: typeof rec.new_str === 'string' ? rec.new_str : '',
    };
  }
  if (rec.command === 'insert') {
    if (typeof rec.insert_line !== 'number' || !Number.isInteger(rec.insert_line)) throw new ParseFault(line);
    if (typeof rec.new_str !== 'string') throw new ParseFault(line);
    return { kind: 'j-insert', path, insertLine: rec.insert_line, newStr: rec.new_str };
  }
  throw new ParseFault(line);
}

function parseJson(patchText: string, lines: string[]): EditDesc[] {
  let firstIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== '') {
      firstIdx = i;
      break;
    }
  }
  if (firstIdx < 0) throw new ParseFault(1);
  const firstChar = lines[firstIdx].trimStart()[0];
  if (firstChar === '[') {
    // One JSON array of edit objects. Any fault is the line the array
    // starts on.
    const startLine = firstIdx + 1;
    let val: unknown;
    try {
      val = JSON.parse(patchText);
    } catch {
      throw new ParseFault(startLine);
    }
    if (!Array.isArray(val)) throw new ParseFault(startLine);
    if (val.length === 0) throw new ParseFault(1);
    return val.map((el) => validateJsonEdit(el, startLine));
  }
  // JSON Lines: one edit object a line, blank lines ignored.
  const edits: EditDesc[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === '') continue;
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch {
      throw new ParseFault(i + 1);
    }
    edits.push(validateJsonEdit(obj, i + 1));
  }
  if (edits.length === 0) throw new ParseFault(1);
  return edits;
}

// ---- SEARCH/REPLACE blocks (SPEC section 7) ----

function parseSearchReplace(lines: string[]): EditDesc[] {
  const edits: EditDesc[] = [];
  let prevEnd = -1; // index of the previous block's >>>>>>> REPLACE line
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (trimEndWs(lines[i]) !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchLine = i + 1;
    // The path is the nearest line before the SEARCH line, after the
    // previous block's end, that is not blank and does not start with
    // three backticks once surrounding whitespace is removed.
    let path: string | null = null;
    for (let j = i - 1; j > prevEnd; j--) {
      const l = lines[j];
      if (l.trim() === '') continue;
      const t = trimWs(l);
      if (t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new ParseFault(searchLine);
      path = prevPath;
    }
    // The ======= line; a >>>>>>> REPLACE (or another SEARCH) before it
    // faults at the SEARCH line.
    let sep = -1;
    let j = i + 1;
    for (; j < lines.length; j++) {
      const t = trimEndWs(lines[j]);
      if (t === '=======') {
        sep = j;
        break;
      }
      if (t === '>>>>>>> REPLACE' || t === '<<<<<<< SEARCH') throw new ParseFault(searchLine);
    }
    if (sep < 0) throw new ParseFault(searchLine);
    let end = -1;
    for (j = sep + 1; j < lines.length; j++) {
      if (trimEndWs(lines[j]) === '>>>>>>> REPLACE') {
        end = j;
        break;
      }
    }
    if (end < 0) throw new ParseFault(searchLine);
    edits.push({
      kind: 's-replace',
      path,
      oldLines: lines.slice(i + 1, sep),
      newLines: lines.slice(sep + 1, end),
    });
    prevPath = path;
    prevEnd = end;
    i = end + 1;
  }
  if (edits.length === 0) throw new ParseFault(1);
  return edits;
}

// ---- V4A patches (SPEC section 8) ----

function parseV4a(lines: string[]): EditDesc[] {
  let first = -1;
  let last = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== '') {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) throw new ParseFault(1);
  if (trimWs(lines[first]) !== '*** Begin Patch') throw new ParseFault(first + 1);
  if (trimWs(lines[last]) !== '*** End Patch') throw new ParseFault(lines.length + 1);
  const edits: EditDesc[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i];
    const lineNo = i + 1;
    if (line.trim() === '') {
      i++;
      continue;
    }
    let kind: 'add' | 'delete' | 'update';
    let path: string;
    if (line.startsWith('*** Add File: ')) {
      kind = 'add';
      path = trimWs(line.slice('*** Add File: '.length));
    } else if (line.startsWith('*** Delete File: ')) {
      kind = 'delete';
      path = trimWs(line.slice('*** Delete File: '.length));
    } else if (line.startsWith('*** Update File: ')) {
      kind = 'update';
      path = trimWs(line.slice('*** Update File: '.length));
    } else {
      throw new ParseFault(lineNo);
    }
    // The section's raw lines run to the next section start (or the end
    // of the patch body). Blank lines at the end of the section are not
    // part of it.
    const bodyStart = i + 1;
    let j = bodyStart;
    while (j < last) {
      const l = lines[j];
      if (
        l.startsWith('*** Add File: ') ||
        l.startsWith('*** Delete File: ') ||
        l.startsWith('*** Update File: ')
      ) {
        break;
      }
      j++;
    }
    const raw = lines.slice(bodyStart, j);
    while (raw.length > 0 && raw[raw.length - 1].trim() === '') raw.pop();
    if (kind === 'add') {
      const fileLines: string[] = [];
      for (let k = 0; k < raw.length; k++) {
        const l = raw[k];
        if (l.startsWith('+')) fileLines.push(l.slice(1));
        else if (l === '') fileLines.push('');
        else throw new ParseFault(bodyStart + k + 1);
      }
      edits.push({ kind: 'v-add', path, lines: fileLines });
    } else if (kind === 'delete') {
      if (raw.length > 0) throw new ParseFault(bodyStart + 1);
      edits.push({ kind: 'u-delete', path });
    } else {
      let moveTo: string | undefined;
      let k = 0;
      if (raw.length > 0 && raw[0].startsWith('*** Move to:')) {
        moveTo = trimWs(raw[0].slice('*** Move to:'.length));
        k = 1;
      }
      const chunks: VChunk[] = [];
      let cur: VChunk | null = null;
      let curO: string[] = [];
      let curN: NewLine[] = [];
      const closeChunk = (): void => {
        if (cur) {
          cur.o = curO;
          cur.n = curN;
          chunks.push(cur);
          cur = null;
          curO = [];
          curN = [];
        }
      };
      for (; k < raw.length; k++) {
        const l = raw[k];
        const ln = bodyStart + k + 1;
        if (l.startsWith('@@')) {
          closeChunk();
          const a = trimWs(l.slice(2));
          cur = { anchor: a === '' ? null : a, o: [], n: [], eof: false };
          curO = [];
          curN = [];
          continue;
        }
        if (l === '*** End of File') {
          if (!cur) throw new ParseFault(ln);
          cur.eof = true;
          closeChunk();
          continue;
        }
        if (l === '' || l[0] === ' ' || l[0] === '-' || l[0] === '+') {
          if (!cur) {
            // Only the first chunk may start at a body line.
            if (chunks.length > 0) throw new ParseFault(ln);
            cur = { anchor: null, o: [], n: [], eof: false };
            curO = [];
            curN = [];
          }
          if (l === '') {
            curO.push('');
            curN.push({ text: '', ctx: curO.length - 1 });
          } else if (l[0] === ' ') {
            const t = l.slice(1);
            curO.push(t);
            curN.push({ text: t, ctx: curO.length - 1 });
          } else if (l[0] === '-') {
            curO.push(l.slice(1));
          } else {
            curN.push({ text: l.slice(1), ctx: null });
          }
          continue;
        }
        throw new ParseFault(ln);
      }
      closeChunk();
      if (chunks.length === 0 && moveTo === undefined) throw new ParseFault(lineNo);
      edits.push({ kind: 'v-update', path, moveTo, chunks });
    }
    i = j;
  }
  if (edits.length === 0) throw new ParseFault(1);
  return edits;
}

// new_str's lines: split at \n, one trailing \r removed from each, without
// the empty piece after a final \n; an empty new_str has none.
function splitNewStrLines(s: string): string[] {
  if (s === '') return [];
  const parts = s.split('\n');
  if (s.endsWith('\n')) parts.pop();
  return parts.map((p) => (p.endsWith('\r') ? p.slice(0, -1) : p));
}

// ---- applying one edit (SPEC section 10) ----

// Every path an edit names is checked for safety before anything else
// about it. A failed edit changes nothing; the edits after it still run.
function applyEdit(desc: EditDesc, strict: boolean, tree: Tree): ApplyOutcome {
  switch (desc.kind) {
    case 'u-noop':
      return okOutcome();
    case 'u-create': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (tree.exists(desc.path)) return failOutcome('exists', desc.path);
      const text =
        desc.plusLines.length === 0
          ? ''
          : desc.plusLines.join('\n') + (desc.noBreak ? '' : '\n');
      tree.set(desc.path, text);
      return okOutcome();
    }
    case 'u-delete': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (!tree.exists(desc.path)) return failOutcome('missing', desc.path);
      tree.set(desc.path, null);
      return okOutcome();
    }
    case 'u-change': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      const curText = tree.get(desc.path);
      if (curText === null) return failOutcome('missing', desc.path);
      const fs = parseFileText(curText);
      if (!applyHunksToState(fs, desc.hunks, strict)) return failOutcome('no-match', desc.path);
      tree.set(desc.path, renderFileText(fs.lines, fs.brk, fs.ends));
      return okOutcome();
    }
    case 'u-move': {
      if (!isSafePath(desc.oldPath)) return failOutcome('unsafe', desc.oldPath);
      if (!isSafePath(desc.newPath)) return failOutcome('unsafe', desc.newPath);
      const curText = tree.get(desc.oldPath);
      if (curText === null) return failOutcome('missing', desc.oldPath);
      if (tree.exists(desc.newPath)) return failOutcome('exists', desc.newPath);
      const fs = parseFileText(curText);
      if (!applyHunksToState(fs, desc.hunks, strict)) return failOutcome('no-match', desc.oldPath);
      tree.set(desc.oldPath, null);
      tree.set(desc.newPath, renderFileText(fs.lines, fs.brk, fs.ends));
      return { failed: null, move: { old: desc.oldPath, next: desc.newPath } };
    }
    case 'j-create': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (tree.exists(desc.path)) return failOutcome('exists', desc.path);
      tree.set(desc.path, desc.text);
      return okOutcome();
    }
    case 'j-str-replace': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      const curText = tree.get(desc.path);
      if (curText === null) return failOutcome('missing', desc.path);
      const crlf = parseFileText(curText).brk === '\r\n';
      let text = curText;
      let oldS = desc.oldStr;
      let newS = desc.newStr;
      if (crlf) {
        text = text.replace(/\r\n/g, '\n');
        oldS = oldS.replace(/\r\n/g, '\n');
        newS = newS.replace(/\r\n/g, '\n');
      }
      // Count occurrences; positions may overlap.
      let count = 0;
      let first = -1;
      let from = 0;
      for (;;) {
        const f = text.indexOf(oldS, from);
        if (f < 0) break;
        count++;
        if (count === 1) first = f;
        from = f + 1;
      }
      if (count === 0) return failOutcome('no-match', desc.path);
      if (count > 1) return failOutcome('ambiguous', desc.path);
      let result = text.slice(0, first) + newS + text.slice(first + oldS.length);
      if (crlf) result = result.replace(/\n/g, '\r\n');
      tree.set(desc.path, result);
      return okOutcome();
    }
    case 'j-insert': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      const curText = tree.get(desc.path);
      if (curText === null) return failOutcome('missing', desc.path);
      const fs = parseFileText(curText);
      if (desc.insertLine < 0 || desc.insertLine > fs.lines.length) {
        return failOutcome('no-match', desc.path);
      }
      const ins = splitNewStrLines(desc.newStr);
      fs.lines.splice(desc.insertLine, 0, ...ins);
      tree.set(desc.path, renderFileText(fs.lines, fs.brk, fs.ends));
      return okOutcome();
    }
    case 's-replace': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (desc.oldLines.length === 0) {
        if (tree.exists(desc.path)) return failOutcome('exists', desc.path);
        const text = desc.newLines.length === 0 ? '' : desc.newLines.join('\n') + '\n';
        tree.set(desc.path, text);
        return okOutcome();
      }
      const curText = tree.get(desc.path);
      if (curText === null) return failOutcome('missing', desc.path);
      const fs = parseFileText(curText);
      const found = findTier(fs.lines, desc.oldLines, strict);
      if (!found) return failOutcome('no-match', desc.path);
      if (found.positions.length > 1) return failOutcome('ambiguous', desc.path);
      const pos = found.positions[0];
      const replacement = buildReplacement(
        fs.lines,
        pos,
        desc.oldLines,
        desc.newLines.map((t) => ({ text: t, ctx: null })),
        found.tier,
      );
      fs.lines.splice(pos, desc.oldLines.length, ...replacement);
      tree.set(desc.path, renderFileText(fs.lines, fs.brk, fs.ends));
      return okOutcome();
    }
    case 'v-add': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (tree.exists(desc.path)) return failOutcome('exists', desc.path);
      const text = desc.lines.length === 0 ? '' : desc.lines.join('\n') + '\n';
      tree.set(desc.path, text);
      return okOutcome();
    }
    case 'v-update': {
      if (!isSafePath(desc.path)) return failOutcome('unsafe', desc.path);
      if (desc.moveTo !== undefined && !isSafePath(desc.moveTo)) {
        return failOutcome('unsafe', desc.moveTo);
      }
      const curText = tree.get(desc.path);
      if (curText === null) return failOutcome('missing', desc.path);
      if (desc.moveTo !== undefined && tree.exists(desc.moveTo)) {
        return failOutcome('exists', desc.moveTo);
      }
      const fs = parseFileText(curText);
      let cursor = 0;
      for (const ch of desc.chunks) {
        if (ch.anchor !== null) {
          const foundA = findTier(fs.lines.slice(cursor), [ch.anchor], strict);
          if (!foundA) return failOutcome('no-match', desc.path);
          cursor = cursor + foundA.positions[0] + 1;
        }
        if (ch.o.length === 0) {
          const insertPos = ch.eof ? fs.lines.length : cursor;
          const ins = ch.n.map((item) => item.text);
          fs.lines.splice(insertPos, 0, ...ins);
          cursor = insertPos + ins.length;
          continue;
        }
        let pos = -1;
        let tier = 0;
        if (ch.eof) {
          // Only an occurrence ending at the file's last line counts, at
          // the first tier with one.
          const tiers = strict ? [0] : [0, 1, 2];
          eofSearch: for (const t of tiers) {
            const occ = findOccurrences(fs.lines.slice(cursor), ch.o, t);
            for (const p of occ) {
              if (cursor + p + ch.o.length === fs.lines.length) {
                pos = cursor + p;
                tier = t;
                break eofSearch;
              }
            }
          }
          if (pos < 0) return failOutcome('no-match', desc.path);
        } else {
          const found = findTier(fs.lines.slice(cursor), ch.o, strict);
          if (!found) return failOutcome('no-match', desc.path);
          pos = cursor + found.positions[0];
          tier = found.tier;
        }
        const replacement = buildReplacement(fs.lines, pos, ch.o, ch.n, tier);
        fs.lines.splice(pos, ch.o.length, ...replacement);
        cursor = pos + replacement.length;
      }
      const outText = renderFileText(fs.lines, fs.brk, fs.ends);
      if (desc.moveTo !== undefined) {
        tree.set(desc.path, null);
        tree.set(desc.moveTo, outText);
        return { failed: null, move: { old: desc.path, next: desc.moveTo } };
      }
      tree.set(desc.path, outText);
      return okOutcome();
    }
  }
}

// ---- format detection (SPEC section 4) ----

function detectFormat(lines: string[]): FormatName | null {
  let firstNonBlank: string | undefined;
  for (const l of lines) {
    if (l.trim() !== '') {
      firstNonBlank = l;
      break;
    }
  }
  if (firstNonBlank !== undefined) {
    if (firstNonBlank.trim() === '*** Begin Patch') return 'v4a';
    const firstChar = firstNonBlank.trimStart()[0];
    if (firstChar === '[' || firstChar === '{') return 'json';
  }
  for (const l of lines) {
    if (l.trimEnd() === '<<<<<<< SEARCH') return 'search-replace';
  }
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('diff --git ')) return 'unified';
    if (lines[i].startsWith('--- ') && i + 1 < lines.length && lines[i + 1].startsWith('+++ ')) {
      return 'unified';
    }
  }
  return null;
}

// ---- invocation (SPEC section 1) ----

interface CliArgs {
  format: FormatName;
  check: boolean;
  strict: boolean;
  patch: string | null;
}

function parseArgs(argv: string[]): CliArgs | null {
  let format: FormatName = 'auto';
  let check = false;
  let strict = false;
  let seenFormat = false;
  let seenCheck = false;
  let seenStrict = false;
  let ended = false;
  const positional: string[] = [];
  for (const arg of argv) {
    if (!ended && arg === '--') {
      ended = true;
      continue;
    }
    if (!ended && arg.startsWith('-') && arg !== '-') {
      if (arg === '--check') {
        if (seenCheck) return null;
        seenCheck = true;
        check = true;
      } else if (arg === '--strict') {
        if (seenStrict) return null;
        seenStrict = true;
        strict = true;
      } else if (arg.startsWith('--format=')) {
        if (seenFormat) return null;
        seenFormat = true;
        const v = arg.slice('--format='.length);
        if (v !== 'unified' && v !== 'search-replace' && v !== 'v4a' && v !== 'json' && v !== 'auto') {
          return null;
        }
        format = v;
      } else {
        return null;
      }
    } else {
      positional.push(arg);
    }
  }
  if (positional.length > 1) return null;
  return { format, check, strict, patch: positional.length === 1 ? positional[0] : null };
}

// ---- main (SPEC sections 2 and 10) ----

function writeLine(line: string): void {
  process.stdout.write(line + '\n');
}

function runCli(): void {
  const argv = process.argv.slice(2);
  const parsed = parseArgs(argv);
  if (parsed === null) {
    writeLine('error: usage');
    process.exitCode = 2;
    return;
  }
  let patchText: string;
  if (parsed.patch === null || parsed.patch === '-') {
    try {
      patchText = readFileSync(0, 'utf8');
    } catch {
      patchText = '';
    }
  } else {
    try {
      patchText = readFileSync(parsed.patch, 'utf8');
    } catch {
      writeLine('error: no-patch');
      process.exitCode = 2;
      return;
    }
  }
  const lines = splitPatchLines(patchText);
  let format: FormatName = parsed.format;
  if (format === 'auto') {
    const detected = detectFormat(lines);
    if (detected === null) {
      writeLine('error: format');
      process.exitCode = 2;
      return;
    }
    format = detected;
  }
  let edits: EditDesc[];
  try {
    edits =
      format === 'json'
        ? parseJson(patchText, lines)
        : format === 'search-replace'
          ? parseSearchReplace(lines)
          : format === 'v4a'
            ? parseV4a(lines)
            : parseUnified(lines);
  } catch (e) {
    if (e instanceof ParseFault) {
      writeLine('error: parse ' + e.line);
      process.exitCode = 2;
      return;
    }
    throw e;
  }
  const tree = new Tree();
  const failures: EditFailure[] = [];
  const moves: { old: string; next: string }[] = [];
  for (const edit of edits) {
    const r = applyEdit(edit, parsed.strict, tree);
    if (r.failed) failures.push(r.failed);
    else if (r.move) moves.push(r.move);
  }
  if (failures.length > 0) {
    const outLines = failures.map((f) => 'error: ' + f.code + ' ' + f.path);
    outLines.push('failed: ' + failures.length + ' of ' + edits.length + ' edits');
    process.stdout.write(outLines.join('\n') + '\n');
    process.exitCode = 1;
    return;
  }
  // What changed, in byte order of the paths (UTF-8). A move that
  // succeeded replaces its pair's D and A lines and sorts by its new path.
  const byPath = new Map<string, { key: string; line: string }>();
  for (const p of tree.touched()) {
    const before = tree.origOf(p);
    const after = tree.get(p);
    if (before === after) continue;
    let line: string;
    if (before === null) line = 'A ' + p;
    else if (after === null) line = 'D ' + p;
    else line = 'M ' + p;
    byPath.set(p, { key: p, line });
  }
  for (const m of moves) {
    const beforeOld = tree.origOf(m.old);
    const afterOld = tree.get(m.old);
    const beforeNew = tree.origOf(m.next);
    const afterNew = tree.get(m.next);
    if (beforeOld !== null && afterOld === null && beforeNew === null && afterNew !== null) {
      byPath.delete(m.old);
      byPath.delete(m.next);
      byPath.set('R:' + m.next, { key: m.next, line: 'R ' + m.old + ' -> ' + m.next });
    }
  }
  const sorted = [...byPath.values()].sort((a, b) =>
    Buffer.compare(Buffer.from(a.key, 'utf8'), Buffer.from(b.key, 'utf8')),
  );
  const outLines = sorted.map((e) => e.line);
  outLines.push('ok: ' + sorted.length + (sorted.length === 1 ? ' file' : ' files'));
  if (!parsed.check) {
    for (const p of tree.touched()) {
      const before = tree.origOf(p);
      const after = tree.get(p);
      if (before === after) continue;
      if (after === null) {
        try {
          unlinkSync(p);
        } catch {
          // The file was already gone.
        }
      } else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, after, 'utf8');
      }
    }
  }
  process.stdout.write(outLines.join('\n') + '\n');
  process.exitCode = 0;
}

// Hand a trial over to the runner when the trial environment is set;
// otherwise run the CLI on the process arguments.
const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv) {
  import('./runner.ts').then(({ runTrial }) => runTrial(trialEnv));
} else {
  runCli();
}
