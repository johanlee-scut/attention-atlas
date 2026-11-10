'use strict';

const tokenNames = ['昨', '夜', '微', '雨', '今', '晨', '放', '晴'];
const tokenCount = tokenNames.length;
const width = 512;
const heads = 8;
const channels = width / heads;
const range = n => Array.from({ length: n }, (_, i) => i);
const indices = range(tokenCount);
const headIds = range(heads);
const dims = range(channels);
const rows = range(4);
const vector = (n, f) => Float64Array.from({ length: n }, (_, i) => f(i));
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const norm = x => {
  const length = Math.sqrt(dot(x, x));
  return x.map(y => y / length);
};
const feature = x => x.map(y => y >= 0 ? y + 1 : Math.exp(y));
const number = x => (Math.abs(x) < .0005 ? 0 : x).toFixed(3);
const ordinal = i => String(i + 1).padStart(2, '0');

let seed = 219199;
function sample() {
  seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
  return (seed / 4294967296 - .5) * Math.sqrt(12);
}

const input = indices.map(() => vector(width, sample));
function weights() {
  return range(width).map(() => vector(width, () => sample() / Math.sqrt(width)));
}
function project(inputRows) {
  const w = weights();
  return inputRows.map(x => vector(width, i => dot(w[i], x)));
}
const query = project(input);
const key = project(input);
const value = project(input);
const outputWeights = weights();
const split = (x, h) => x.subarray(h * channels, (h + 1) * channels);
const headData = headIds.map(h => ({
  q: query.map(x => split(x, h)),
  k: key.map(x => split(x, h)),
  v: value.map(x => split(x, h)),
}));

function attention(h) {
  const { q, k, v } = headData[h];
  const scores = indices.map(t => indices.map(i => i <= t ? dot(q[t], k[i]) / Math.sqrt(channels) : null));
  const weights = scores.map((row, t) => {
    const max = Math.max(...row.slice(0, t + 1));
    const exp = row.slice(0, t + 1).map(score => Math.exp(score - max));
    const total = exp.reduce((sum, x) => sum + x, 0);
    return row.map((_, i) => i <= t ? exp[i] / total : null);
  });
  const out = weights.map((row, t) => vector(channels, j =>
    row.slice(0, t + 1).reduce((sum, w, i) => sum + w * v[i][j], 0)));
  return { scores, weights, out };
}

const soft = headIds.map(attention);
const joined = indices.map(t => Float64Array.from(soft.flatMap(h => [...h.out[t]])));
const multiOut = joined.map(x => vector(width, i => dot(outputWeights[i], x)));

const kernels = headIds.map(h => {
  const { q, k, v } = headData[h];
  const qf = q.map(feature);
  const kf = k.map(feature);
  const linear = [];
  let S = dims.map(() => new Float64Array(channels));
  let z = new Float64Array(channels);
  for (const t of indices) {
    const prev = S;
    const kNow = kf[t];
    S = dims.map(i => S[i].map((x, j) => x + kNow[i] * v[t][j]));
    z = z.map((x, i) => x + kNow[i]);
    const denom = dot(qf[t], z);
    const out = vector(channels, j => dims.reduce((sum, i) => sum + qf[t][i] * S[i][j], 0) / denom);
    const affinities = kf.slice(0, t + 1).map(x => dot(qf[t], x));
    const directDenom = affinities.reduce((sum, x) => sum + x, 0);
    const direct = vector(channels, j => affinities.reduce((sum, a, i) => sum + a * v[i][j], 0) / directDenom);
    const error = Math.max(...out.map((x, j) => Math.abs(x - direct[j])));
    linear.push({ prev, S, z, denom, out, direct, error });
  }
  return { qf, kf, linear };
});

function deltaHistory(qHead, vHead, gate, beta, until) {
  const q = headData[qHead].q.map(norm);
  const k = headData[qHead].k.map(norm);
  const v = headData[vHead].v;
  let S = dims.map(() => new Float64Array(channels));
  let frame;
  for (let t = 0; t <= until; t++) {
    const prev = S;
    const decayed = dims.map(i => prev[i].map(x => x * gate[i]));
    const recall = vector(channels, j => dims.reduce((sum, i) => sum + decayed[i][j] * k[t][i], 0));
    const residual = v[t].map((x, i) => x - recall[i]);
    S = dims.map(i => decayed[i].map((x, j) => x + beta * k[t][i] * residual[j]));
    const out = vector(channels, j => dims.reduce((sum, i) => sum + S[i][j] * q[t][i], 0));
    frame = { prev, decayed, recall, residual, S, out };
  }
  return frame;
}

const lessons = [
  {
    id: 'softmax', tab: '普通 Attention', title: '逐对比较，再加权', type: '因果 Softmax',
    try: '点选第 8 个 token，再回到第 2 个。未来 token 被遮住，当前行的有效权重始终加起来等于 1。',
    phases: [
      ['打分', '当前 token 和历史做点积', '每一行代表一个 query，每一列代表一个 key。只计算截至当前 token 的因果前缀。除以 √64，控制点积的尺度。', 'aₜⱼ = qₜᵀ kⱼ / √64，j ≤ t'],
      ['归一化', 'Softmax 把分数变成权重', '只对该行已可见的 key 做指数归一化。未来位置的权重为零，每行有效权重的总和为 1。', 'wₜⱼ = exp(aₜⱼ) / Σᵢ≤ₜ exp(aₜᵢ)'],
      ['读取', '按权重混合历史 value', '每一项的权重乘到整条 64 维 value 上，然后沿 token 维求和。图中右侧显示结果的前 4 维。', 'oₜ = Σⱼ≤ₜ wₜⱼ vⱼ'],
    ],
  },
  {
    id: 'mha', tab: 'MHA', title: '八个头，各自看一遍', type: '多头布局',
    try: '在头部选择器中切换 H1 和 H2，比较同一 token 的权重图。两者来自不同的投影子空间。',
    phases: [
      ['拆头', '从 512 维投影出 8 个头', 'Q、K、V 分别由同一份输入线性投影。每份 512 维投影沿最后一维拆成 8 个 64 维头。', 'XW_Q, XW_K, XW_V → [8 token, 8 head, 64]'],
      ['并行', '每个头有自己的因果权重', '每个头独立做一次缩放点积和 Softmax。选择任意头，查看它的权重图和读取结果。', 'oₜʰ = Σⱼ≤ₜ softmaxⱼ(qₜʰᵀ kⱼʰ / √64) vⱼʰ'],
      ['拼接', '合并 8 个输出再投影', '按头顺序拼接，恢复 512 维。Wₒ 混合不同头的输出，产生该 token 的最终表示。', 'oₜ = Wₒ · concat(oₜ¹, …, oₜ⁸)'],
    ],
  },
  {
    id: 'linear', tab: '线性 Attention', title: '把历史折叠成状态', type: '核化递推',
    try: '切到「读取」，观察递推结果与显式核权重结果的最大误差。再切换 token，状态的形状仍然相同。',
    phases: [
      ['换核', '用可分解的相似度取代 Softmax 核', '用正特征 φ(x) = ELU(x) + 1，使相似度能写成 query 特征和 key 特征的内积。下面的结合律对这个核精确成立。', 'κ(q,k) = φ(q)ᵀ φ(k)'],
      ['积累', '先写入 K 与 V 的外积', '当前 key 和 value 的外积加进 S，key 特征也加进 z。按顺序写入保证因果性。固定头宽时，整段的计算量随 token 数线性增长。', 'Sₜ = Sₜ₋₁ + φ(kₜ) vₜᵀ，zₜ = zₜ₋₁ + φ(kₜ)'],
      ['读取', '把 Query 放到求和之外', '把 query 的特征乘到 S 和 z 上，分别得到分子与归一化分母。显式求和与递推得到同样的核化输出。它通常与普通 Softmax Attention 的输出不同。', 'oₜ = Sₜᵀ φ(qₜ) / (φ(qₜ)ᵀ zₜ)'],
    ],
  },
  {
    id: 'delta', tab: 'Delta Rule', title: '先查旧值，再写差值', type: '误差修正',
    try: '把 β 调到 0，观察写入消失。再设为 1，当前单位长度 key 的读回值会等于目标 value。',
    phases: [
      ['预测', '让当前 Key 查一下旧状态', '先问旧状态：沿这个 key 方向已经记住了什么？图中先用 key 读取。最后一步才会用 query 产生输出。', 'v̂ₜ = Sₜ₋₁ᵀ kₜ，‖kₜ‖₂ = 1'],
      ['纠偏', '写入目标与旧值之差', '普通累加不检查旧状态。Delta Rule 计算目标和旧值的残差，只沿当前 key 方向按 β 写入修正。', 'rₜ = vₜ − v̂ₜ；Sₜ = Sₜ₋₁ + β kₜ rₜᵀ'],
      ['读取', '用 Query 读更新后的状态', '读回输出来自更新后的 S。对当前 key 而言，β=1 时修正量正好补足原来的误差。', 'oₜ = Sₜᵀ qₜ'],
    ],
  },
  {
    id: 'gdn', tab: 'GDN', title: '整头遗忘，再做 Delta', type: '标量遗忘门',
    try: '切到后几个 token，把 α 从 1 拖到 0。看历史状态如何一起缩小，再观察本轮写入。',
    phases: [
      ['遗忘', '一个 α 缩放整个头的旧状态', 'Gated DeltaNet 在误差修正前先衰减。每个头、每个 token 可以有自己的标量门。这里用滑杆给选中头的所有 token 设同一个值。', 'Ŝₜ = α Sₜ₋₁'],
      ['纠偏', '按衰减后的记忆计算残差', '写入仍然使用 Delta Rule。预测必须从已经衰减的状态读取，否则会用错修正目标。', 'rₜ = vₜ − Ŝₜᵀ kₜ；Sₜ = Ŝₜ + β kₜ rₜᵀ'],
      ['读取', '遗忘与写入各有一个旋钮', 'α 控制旧状态保留多少，β 控制这次修正写入多少。α=1 时退回纯 Delta Rule。', 'oₜ = Sₜᵀ qₜ'],
    ],
  },
  {
    id: 'gva', tab: 'GVA', title: '共享 Q/K，分开存 Value', type: 'Value 头分组',
    try: '交替点 V1 和 V2。它们使用同一组 Q/K，却有不同的状态和输出。',
    phases: [
      ['分组', '4 组 Q/K 驱动 8 个 Value 头', '每组 Q/K 供两个 value 头复用。以 Qwen3-Next GDN 的头部布局为例，这里缩成 4:8，每个头仍为 64 维。', 'g(h) = ⌊h / 2⌋，h ∈ {0,…,7}'],
      ['写入', '每个 Value 头维护自己的 S', '同组两个 value 头使用同一个 key，但各自的 value 和旧状态不同。因此它们的 Delta 写入及递推状态可以不同。', 'Sₜʰ = GDN(Sₜ₋₁ʰ, kₜᵍ⁽ʰ⁾, vₜʰ)'],
      ['读取', '同一个 Query，读出两路内容', '同组内用相同的 query 读两份独立状态。此处复用上一章的 GDN 更新，改变的是 Q/K 到 value 头的连接方式。', 'oₜʰ = (Sₜʰ)ᵀ qₜᵍ⁽ʰ⁾'],
    ],
  },
  {
    id: 'kda', tab: 'Kimi KDA', title: '把遗忘细化到每个通道', type: '逐通道遗忘门',
    try: '把通道 1 调到 1、通道 2 调到 0，观察衰减矩阵的前两行。其余通道固定 0.85。',
    phases: [
      ['逐行遗忘', '一个头里，每个 Key 通道各有门值', 'KDA 把 α 变成长度 64 的向量。由于状态行对应 key 通道，左乘对角矩阵相当于逐行缩放。', 'Ŝₜ = diag(α₁, …, α₆₄) Sₜ₋₁'],
      ['纠偏', '在逐通道衰减之后做 Delta', '仍然沿当前 key 方向写入误差修正。选择性遗忘和定向写入可以同时作用于同一份状态。', 'Sₜ = Ŝₜ + β kₜ (vₜ − Ŝₜᵀ kₜ)ᵀ'],
      ['读取', '相等的门值还原标量衰减', '把两个滑杆都设为 0.85，此时所有通道门值相同。固定输入、β 和 α=0.85 时，演示的 KDA 递推与 GDN 一致。', 'oₜ = Sₜᵀ qₜ，diag(α, …, α) = αI'],
    ],
  },
];

const el = id => document.getElementById(id);
let chapter = 0;
let position = 0;
let head = 0;
let alpha = .85;
let beta = .8;
let alphaFirst = .95;
let alphaSecond = .25;
let timer = null;
const maxPosition = tokenCount * 3 - 1;

function matrix(label, size, data, options = {}) {
  const scale = Math.max(.001, ...data.flatMap(row => [...row].filter(x => x !== null).map(Math.abs)));
  const cells = data.map((row, i) => [...row].map((x, j) => {
    if (x === null) return '<span class="cell mask" aria-label="遮罩或尚未处理">·</span>';
    const tone = x < 0 ? '201, 179, 252' : (options.tone || '140, 221, 244');
    const opacity = .09 + .42 * Math.min(1, Math.abs(x) / scale);
    const highlight = options.row === i ? ' focus' : '';
    const diagonal = options.diag && i === j ? ' diag' : '';
    return '<span class="cell' + highlight + diagonal + '" style="--cell: rgba(' + tone + ',' + opacity + ')">' + number(x) + '</span>';
  }).join('')).join('');
  return '<div class="matrix ' + (options.compact ? 'compact' : '') + '" role="group" aria-label="' + label + ' ' + size + '">' +
    '<div class="matrix-head"><strong>' + label + '</strong><span>' + size + '</span></div>' +
    '<div class="matrix-grid" style="--cols:' + data[0].length + '">' + cells + '</div></div>';
}

const tile = (name, state) => matrix(name, '64×64 · 前4行列', rows.map(i => state[i].slice(0, 4)));
const strip = (name, data, n = channels) => matrix(name, n + ' 维 · 前4项', [data.slice(0, 4)]);
const arrow = glyph => '<div class="arrow" aria-hidden="true">' + (glyph || '→') + '</div>';
function stage(title, right, contents, caption, animate) {
  el('stage').innerHTML = '<div class="stage-title"><strong>' + title + '</strong><span>' + right + '</span></div>' +
    '<div class="visuals wrap' + (animate ? ' animate' : '') + '">' + contents + '</div>' +
    '<p class="caption">' + caption + '</p>';
}

function causalGrid(data, t, label) {
  return matrix(label, '8×8 · Q ↓ K →', data.map((row, i) => i <= t ? row : indices.map(() => null)), {
    row: t, compact: true, tone: '200, 243, 151',
  });
}

function meters(weights, t) {
  return '<div class="meter-stack" aria-label="当前行归一化权重">' + weights.slice(0, t + 1).map((w, i) =>
    '<div class="meter"><span>K' + (i + 1) + '</span><i aria-hidden="true"><span style="--amount:' + (w * 100) +
    '%"></span></i><span>' + number(w) + '</span></div>').join('') + '</div>';
}

function softScene(t, phase, animate, h = 0) {
  const { scores, weights, out } = soft[h];
  const grid = causalGrid(phase === 0 ? scores : weights, t, phase === 0 ? '因果分数' : '因果权重');
  const result = phase === 0 ? strip('当前 q', headData[h].q[t]) :
    phase === 1 ? meters(weights[t], t) : strip('当前输出 o', out[t]);
  const caption = phase === 0 ? '高亮行是当前 query。每个未遮挡格子对应一次 q · k。' :
    '点号表示未来位置或尚未处理的行。当前行有效权重之和 = ' + number(weights[t].slice(0, t + 1).reduce((s, x) => s + x, 0)) + '。';
  stage('TOKEN ' + ordinal(t) + ' · ' + tokenNames[t], 'HEAD ' + (h + 1), grid + arrow() + result, caption, animate);
}

function routes(grouped) {
  const groupSize = grouped ? 2 : 1;
  const route = range(heads / groupSize).map(g => {
    const branches = range(groupSize).map(i => {
      const h = g * groupSize + i;
      return '<div class="flow-row' + (h === head ? ' active' : '') + '">' +
        '<button type="button" data-head="' + h + '" aria-pressed="' + (h === head) +
        '">V' + (h + 1) + '</button><span class="route-s">' + (grouped ? '状态 S' : '注意力 A') + (h + 1) +
        '</span><span class="route-o">输出 o' + (h + 1) + ' · 64维</span></div>';
    }).join('');
    return '<div class="route-group"><span class="route-q">Q' + (g + 1) + ' / K' + (g + 1) +
      '</span><div class="branches">' + branches + '</div></div>';
  }).join('');
  return '<div class="flow">' + route + '</div>';
}

function multiScene(t, phase, animate) {
  if (phase === 1) {
    softScene(t, 2, animate, head);
    return;
  }
  if (phase === 0) {
    stage('8 路独立计算', '8 × 64 = 512', routes(false), '点击任一 V 头，选中这路 Q/K/V。', animate);
    return;
  }
  const samples = headIds.map(h => soft[h].out[t].slice(0, 1));
  stage('拼接 → 输出投影', '512 → 512', matrix('每头第 1 项', '8 个头', samples) + arrow() +
    strip('concat', joined[t], width) + arrow() + strip('Wₒ · concat', multiOut[t], width),
    '拼接保留完整的 8×64 个数。左侧从每个头取一个数作索引，右侧只展示前 4 维。', animate);
}

function linearScene(t, phase, animate) {
  const { qf, kf, linear } = kernels[head];
  const f = linear[t];
  if (phase === 0) {
    stage('Φ(Q) · Φ(K)ᵀ', '每个向量 64 维', strip('当前 q', headData[head].q[t]) + arrow() +
      strip('φ(q)', qf[t]) + strip('φ(k)', kf[t]), '正特征让相似度可分解，也能保留归一化分母。', animate);
  } else if (phase === 1) {
    stage('写入固定形状的状态', '64×64 + 64', tile('旧状态 S', f.prev) + arrow() + tile('新状态 S', f.S) +
      strip('累计 z', f.z), 'S 沿 token 维累加外积。即使序列超过 8 个 token，状态仍是 64×64 加 64。', animate);
  } else {
    stage('两种计算路径', '分母 ' + number(f.denom),
      '<div class="compare"><small>先算 KᵀV，再由 Q 读取</small>' + strip('递推读取', f.out) + '</div>' + arrow('≈') +
      '<div class="compare"><small>先算 QKᵀ，再乘 V</small>' + strip('显式核求和', f.direct) + '</div>',
      '完整 64 维最大绝对差 = ' + f.error.toExponential(2) + '。显示差异来自浮点计算顺序。', animate);
  }
}

function deltaScene(t, phase, mode, animate) {
  const gate = vector(channels, i => {
    if (mode === 'delta') return 1;
    if (mode !== 'kda') return alpha;
    return i === 0 ? alphaFirst : i === 1 ? alphaSecond : .85;
  });
  const f = deltaHistory(head, head, gate, beta, t);
  if (phase === 0 && mode !== 'delta') {
    const title = mode === 'kda' ? '逐行衰减' : '整头衰减';
    const label = mode === 'kda' ? 'α · 通道 1–4' : 'α';
    const gateValues = mode === 'kda' ? [gate.slice(0, 4)] : [[gate[0]]];
    stage(title, 'TOKEN ' + ordinal(t), tile('旧状态 S', f.prev) + arrow('×') +
      matrix(label, '', gateValues, { tone: '255, 212, 135' }) + arrow() + tile('衰减后 Ŝ', f.decayed),
      '矩阵行是 Key 通道，列是 Value 通道。左侧和右侧均展示同一位置的 4×4 切片。', animate);
  } else if (phase === 0) {
    stage('读旧值', 'TOKEN ' + ordinal(t), tile('旧状态 S', f.prev) + arrow() + strip('读回预测 v̂', f.recall),
      '预测是当前 key 对旧状态的读回值。', animate);
  } else if (phase === 1) {
    stage('计算误差，再写入', 'β = ' + number(beta), strip('目标 v', headData[head].v[t]) + arrow('−') +
      strip('读回 v̂', f.recall) + arrow('=') + strip('残差 r', f.residual),
      '状态改变量是 β × 当前 key 与残差的外积。下一步查看更新后状态。', animate);
  } else {
    stage('新状态 → Query 读取', 'TOKEN ' + ordinal(t), tile('更新后 S', f.S) + arrow() + strip('输出 o', f.out),
      '本 token 的输出读自更新后的状态。向前一步，就能看到这个状态成为下一 token 的历史。', animate);
  }
}

function groupedScene(t, phase, animate) {
  const group = Math.floor(head / 2);
  const pair = [group * 2, group * 2 + 1];
  if (phase === 0) {
    stage('两路 Value 共享一组 Q/K', '4 组 Q/K · 8 个 V', routes(true),
      '每两行使用相同编号的 Q/K，各自连接一个 value 头、状态和输出。', animate);
    return;
  }
  const gate = vector(channels, () => alpha);
  const frames = pair.map(h => deltaHistory(group, h, gate, beta, t));
  const contents = frames.map((f, i) => {
    const name = (pair[i] === head ? '● ' : '') + (pair[i] + 1);
    return phase === 1 ? tile('S ' + name, f.S) : strip('o ' + name, f.out);
  }).join('<span class="arrow" aria-hidden="true">∥</span>');
  stage('同组双路对比', 'Q' + (group + 1) + ' / K' + (group + 1), contents,
    '同一个 token、同一对 Q/K。两路使用各自的 V，维护各自的 64×64 状态。', animate);
}

const renderers = {
  softmax: (t, p, a) => softScene(t, p, a),
  mha: multiScene,
  linear: linearScene,
  delta: (t, p, a) => deltaScene(t, p, 'delta', a),
  gdn: (t, p, a) => deltaScene(t, p, 'gdn', a),
  gva: groupedScene,
  kda: (t, p, a) => deltaScene(t, p, 'kda', a),
};

const settings = {
  alpha: ['保留系数 α', () => alpha, x => { alpha = x; }],
  beta: ['写入系数 β', () => beta, x => { beta = x; }],
  first: ['通道 1 · α₁', () => alphaFirst, x => { alphaFirst = x; }],
  second: ['通道 2 · α₂', () => alphaSecond, x => { alphaSecond = x; }],
};
const knobs = { delta: ['beta'], gdn: ['alpha', 'beta'], kda: ['first', 'second', 'beta'] };

function headPicker(id) {
  const label = id === 'gva' ? 'Value 头' : '选择注意力头';
  return '<p class="settings-label">' + label + '</p><div class="head-buttons" role="group" aria-label="' + label + '">' +
    headIds.map(h => '<button type="button" data-head="' + h + '" aria-pressed="' + (h === head) + '">' +
      (id === 'gva' ? 'V' : 'H') + (h + 1) + '</button>').join('') + '</div>';
}

function refreshSettings() {
  const { id } = lessons[chapter];
  const sliders = (knobs[id] || []).map(key => {
    const [label, get] = settings[key];
    return '<div class="parameter"><label for="param-' + key + '">' + label + '<output for="param-' + key + '">' +
      get().toFixed(2) + '</output></label><input type="range" min="0" max="1" step=".05" value="' + get() +
      '" id="param-' + key + '" data-param="' + key + '"></div>';
  }).join('');
  el('settings').innerHTML = (id !== 'softmax' ? headPicker(id) : '') + sliders;
}

function draw(animate = true) {
  const lesson = lessons[chapter];
  const t = Math.floor(position / 3);
  const phase = position % 3;
  const [, title, detail, formula] = lesson.phases[phase];
  el('chapter-id').textContent = 'CHAPTER ' + ordinal(chapter);
  el('lesson-title').textContent = lesson.title;
  el('chapter-type').textContent = lesson.type;
  el('step-title').textContent = title;
  el('explanation').textContent = detail;
  el('equation').textContent = formula;
  el('prompt').textContent = lesson.try;
  el('advance').textContent = chapter === lessons.length - 1 ? '↺ 回到普通 Attention' : '下一章 · ' + lessons[chapter + 1].tab + ' →';
  el('progress').value = position;
  el('progress').setAttribute('aria-valuetext', 'Token ' + (t + 1) + '，' + lesson.phases[phase][0]);
  el('counter').textContent = ordinal(position) + ' / ' + (maxPosition + 1);
  el('prev').disabled = position === 0;
  el('next').disabled = position === maxPosition;
  const play = el('play');
  play.textContent = timer === null ? '▶ 播放' : 'Ⅱ 暂停';
  play.setAttribute('aria-label', timer === null ? '播放当前章节' : '暂停当前章节');
  [...el('chapters').children].forEach((button, i) => {
    if (i === chapter) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  [...el('tokens').children].forEach((button, i) => {
    button.setAttribute('aria-pressed', String(i === t));
    button.classList.toggle('future', i > t);
  });
  el('step-picker').innerHTML = lesson.phases.map(([label], i) => '<button type="button" data-phase="' + i +
    '" aria-pressed="' + (i === phase) + '">' + ordinal(i) + ' · ' + label + '</button>').join('');
  renderers[lesson.id](t, phase, animate);
}

function stop() {
  clearInterval(timer);
  timer = null;
}
function go(pos) {
  stop();
  position = Math.min(maxPosition, Math.max(0, pos));
  draw();
}
function selectChapter(n) {
  stop();
  chapter = n;
  position = 0;
  head = 0;
  refreshSettings();
  draw();
}

el('chapters').innerHTML = lessons.map((l, i) => '<button type="button" class="chapter" data-chapter="' + i +
  '"><span>' + ordinal(i) + '</span><strong>' + l.tab + '</strong></button>').join('');
el('tokens').innerHTML = tokenNames.map((name, i) => '<button type="button" class="token" data-token="' + i +
  '" aria-label="Token ' + (i + 1) + '，' + name + '"><span>' + ordinal(i) + '</span>' + name + '</button>').join('');
el('progress').max = maxPosition;
el('prev').onclick = () => go(position - 1);
el('next').onclick = () => go(position + 1);
el('advance').onclick = () => selectChapter((chapter + 1) % lessons.length);
el('progress').oninput = event => go(Number(event.target.value));
el('play').onclick = () => {
  if (timer !== null) {
    stop();
  } else {
    if (position === maxPosition) position = 0;
    timer = setInterval(() => {
      position++;
      if (position === maxPosition) stop();
      draw();
    }, 1150);
  }
  draw();
};

document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  const data = button.dataset;
  if (data.chapter !== undefined) selectChapter(Number(data.chapter));
  if (data.token !== undefined) go(Number(data.token) * 3 + position % 3);
  if (data.phase !== undefined) {
    go(position - position % 3 + Number(data.phase));
    el('step-picker').children[Number(data.phase)].focus();
  }
  if (data.head !== undefined) {
    stop();
    head = Number(data.head);
    el('settings').querySelectorAll('[data-head]').forEach(btn =>
      btn.setAttribute('aria-pressed', String(Number(btn.dataset.head) === head)));
    draw();
    if (button.closest('#stage')) el('stage').querySelector('[data-head="' + head + '"]').focus();
  }
});

el('settings').addEventListener('input', event => {
  const param = event.target.dataset.param;
  if (!param) return;
  stop();
  const value = Number(event.target.value);
  settings[param][2](value);
  event.target.parentElement.querySelector('output').textContent = value.toFixed(2);
  draw(false);
});

document.querySelector('.mast-right a').addEventListener('click', () => {
  el('sources').open = true;
});

refreshSettings();
draw(false);
