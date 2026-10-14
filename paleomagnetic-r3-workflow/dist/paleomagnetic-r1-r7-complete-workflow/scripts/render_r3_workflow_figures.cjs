#!/usr/bin/env node

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const figureDirectory = path.join(root, 'docs', 'figures');
const moduleSearchPaths = [
  path.join(root, 'node_modules'),
  path.join(
    os.homedir(),
    '.cache',
    'codex-runtimes',
    'codex-primary-runtime',
    'dependencies',
    'node',
    'node_modules',
  ),
];
const sharp = require(require.resolve('sharp', { paths: moduleSearchPaths }));

const colors = {
  background: '#FFFFFF',
  text: '#17212B',
  muted: '#4B5563',
  line: '#7A8691',
  border: '#667085',
  band: '#CAD5DF',
  input: '#E7F0F7',
  parsing: '#E8F1E7',
  cache: '#DDEFE5',
  knowledge: '#F3EAD5',
  retrieval: '#DCEAF7',
  binding: '#F5E8CC',
  reasoning: '#EEE4F3',
  conflict: '#F5E1D9',
  conflictBorder: '#A6664C',
  neutral: '#E8ECEF',
  output: '#DFECEB',
  outputBorder: '#4D7774',
  label: '#ECE8EF',
  labelBorder: '#75677D',
};

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function bandMarkup(band) {
  return `
    <g class="phase-band">
      <rect x="${band.x}" y="${band.y}" width="${band.width}" height="${band.height}" rx="18" />
      <text x="${band.x + 28}" y="${band.y + 33}" class="phase-label">${escapeXml(band.label)}</text>
    </g>`;
}

function nodeMarkup(node) {
  const stroke = node.stroke || colors.border;
  const fill = node.fill || colors.neutral;
  const shape = node.shape === 'diamond'
    ? `<polygon points="${node.x + node.width / 2},${node.y} ${node.x + node.width},${node.y + node.height / 2} ${node.x + node.width / 2},${node.y + node.height} ${node.x},${node.y + node.height / 2}" fill="${fill}" stroke="${stroke}" stroke-width="2" />`
    : `<rect x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}" rx="12" fill="${fill}" stroke="${stroke}" stroke-width="2" />`;
  const lineHeight = node.lineHeight || 25;
  const firstY = node.y + node.height / 2 - ((node.lines.length - 1) * lineHeight) / 2;
  const text = node.lines.map((line, index) => (
    `<tspan x="${node.x + node.width / 2}" y="${firstY + index * lineHeight}">${escapeXml(line)}</tspan>`
  )).join('');
  return `
    <g class="workflow-node" data-node="${escapeXml(node.id)}">
      ${shape}
      <text class="node-label">${text}</text>
    </g>`;
}

function edgeMarkup(edge) {
  const path = edge.points.map((point, index) => `${index ? 'L' : 'M'} ${point[0]} ${point[1]}`).join(' ');
  const classNames = ['workflow-edge'];
  if (edge.dashed) classNames.push('dashed');
  if (edge.arrow === false) classNames.push('no-arrow');
  const label = edge.label
    ? `<g class="edge-label-group">
        <rect x="${edge.labelX - edge.labelWidth / 2}" y="${edge.labelY - 18}" width="${edge.labelWidth}" height="24" rx="4" />
        <text x="${edge.labelX}" y="${edge.labelY}" class="edge-label">${escapeXml(edge.label)}</text>
      </g>`
    : '';
  return `<g><path d="${path}" class="${classNames.join(' ')}" />${label}</g>`;
}

function junctionMarkup(point) {
  return `<circle cx="${point[0]}" cy="${point[1]}" r="4.2" class="junction" />`;
}

function validateFigure(spec) {
  const junctions = new Set((spec.junctions || []).map(([x, y]) => `${x},${y}`));
  const segments = [];
  spec.edges.forEach((edge, edgeIndex) => {
    edge.points.slice(1).forEach((end, pointIndex) => {
      const start = edge.points[pointIndex];
      if (start[0] !== end[0] && start[1] !== end[1]) {
        throw new Error(`${spec.title}: edge ${edgeIndex} contains a non-orthogonal segment`);
      }
      segments.push({ edgeIndex, start, end });
    });
  });

  const between = (value, first, second) => (
    value >= Math.min(first, second) && value <= Math.max(first, second)
  );
  const endpoint = (segment, x, y) => (
    (segment.start[0] === x && segment.start[1] === y)
    || (segment.end[0] === x && segment.end[1] === y)
  );
  for (let firstIndex = 0; firstIndex < segments.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < segments.length; secondIndex += 1) {
      const first = segments[firstIndex];
      const second = segments[secondIndex];
      if (first.edgeIndex === second.edgeIndex) continue;
      const firstHorizontal = first.start[1] === first.end[1];
      const secondHorizontal = second.start[1] === second.end[1];
      if (firstHorizontal === secondHorizontal) continue;
      const horizontal = firstHorizontal ? first : second;
      const vertical = firstHorizontal ? second : first;
      const x = vertical.start[0];
      const y = horizontal.start[1];
      if (!between(x, horizontal.start[0], horizontal.end[0])) continue;
      if (!between(y, vertical.start[1], vertical.end[1])) continue;
      const allowed = junctions.has(`${x},${y}`)
        || (endpoint(horizontal, x, y) && endpoint(vertical, x, y));
      if (!allowed) {
        throw new Error(
          `${spec.title}: edges ${first.edgeIndex} and ${second.edgeIndex} cross at ${x},${y}`,
        );
      }
    }
  }
}

function figureSvg(spec) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${spec.width}" height="${spec.height}" viewBox="0 0 ${spec.width} ${spec.height}">
  <title>${escapeXml(spec.title)}</title>
  <desc>${escapeXml(spec.description)}</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="${colors.line}" />
    </marker>
  </defs>
  <style>
    text { font-family: "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill: ${colors.text}; }
    .figure-title { font-size: 34px; font-weight: 500; text-anchor: middle; }
    .phase-band rect { fill: none; stroke: ${colors.band}; stroke-width: 2; stroke-dasharray: 8 6; }
    .phase-label { font-size: 22px; font-weight: 500; }
    .node-label { font-size: 20px; font-weight: 400; text-anchor: middle; dominant-baseline: middle; }
    .workflow-edge { fill: none; stroke: ${colors.line}; stroke-width: 2.2; marker-end: url(#arrow); stroke-linejoin: round; }
    .workflow-edge.dashed { stroke-dasharray: 8 6; }
    .workflow-edge.no-arrow { marker-end: none; }
    .junction { fill: ${colors.line}; }
    .edge-label-group rect { fill: ${colors.background}; }
    .edge-label { font-size: 16px; font-weight: 400; fill: ${colors.muted}; text-anchor: middle; }
  </style>
  <rect width="100%" height="100%" fill="${colors.background}" />
  <text x="${spec.width / 2}" y="48" class="figure-title">${escapeXml(spec.title)}</text>
  ${spec.bands.map(bandMarkup).join('')}
  <g class="edges">${spec.edges.map(edgeMarkup).join('')}</g>
  <g class="junctions">${(spec.junctions || []).map(junctionMarkup).join('')}</g>
  <g class="nodes">${spec.nodes.map(nodeMarkup).join('')}</g>
</svg>`;
}

function originalEvidenceFigure() {
  const nodes = [
    { id: 'identity', x: 100, y: 125, width: 430, height: 65, lines: ['目标身份清单', 'REFNO / RESULTNO / ROCKNAME'], fill: colors.input },
    { id: 'papers', x: 685, y: 125, width: 430, height: 65, lines: ['本地原始文献', 'PDF / 已解析页文本'], fill: colors.input },
    { id: 'knowledge', x: 1270, y: 125, width: 430, height: 65, lines: ['冻结 R3 规范与专家知识', 'criterion / 专家规则 / 术语模式'], fill: colors.knowledge },

    { id: 'source', x: 80, y: 300, width: 250, height: 80, lines: ['原文来源检查', '定位 PDF / 已解析文本', '缺失与可用性标记'], fill: colors.neutral },
    { id: 'base', x: 370, y: 300, width: 280, height: 80, lines: ['基础解析候选生成', '已有文本 / pypdf', 'pdfplumber / pdftotext'], fill: colors.parsing },
    { id: 'quality', x: 700, y: 285, width: 200, height: 110, lines: ['基础候选', '存在质量风险?'], fill: colors.neutral, shape: 'diamond' },
    { id: 'mineru', x: 820, y: 430, width: 300, height: 70, lines: ['可选 MinerU 恢复候选', '仅由无标签质量门触发'], fill: colors.parsing },
    { id: 'arbitration', x: 1030, y: 300, width: 300, height: 80, lines: ['候选质量仲裁', '整篇单源选择；必要时逐页融合', '禁止跨源拼接同一页文本'], fill: colors.cache },
    { id: 'cache', x: 1410, y: 300, width: 300, height: 80, lines: ['页级原文缓存', '页码 / 章节提示 / 解析来源', '文本哈希与质量标志'], fill: colors.cache },

    { id: 'target', x: 70, y: 620, width: 290, height: 80, lines: ['目标名称规范化', '目标特异词与别名', '同文献竞争目标集合'], fill: colors.binding },
    { id: 'retrieval', x: 400, y: 620, width: 290, height: 80, lines: ['R3 冻结模式检索', '方法 / 诊断 / 载体', '载体—剩磁桥接'], fill: colors.retrieval },
    { id: 'filter', x: 730, y: 620, width: 290, height: 80, lines: ['候选证据过滤', '参考文献 / 外部引文 / 否定', '未实施方法 / 非结果语句'], fill: colors.retrieval },
    { id: 'scope', x: 1060, y: 620, width: 300, height: 80, lines: ['局部语义作用域', '句子 / 段落 / 小节', '表格行 / 物理页'], fill: colors.binding },
    { id: 'binding', x: 1400, y: 620, width: 300, height: 80, lines: ['证据目标归属', '竞争目标排除', 'scope_id / 绑定强度'], fill: colors.binding },

    { id: 'relations', x: 620, y: 830, width: 560, height: 70, lines: ['目标绑定的规则关系集合', 'SUPPORT / CONFLICT / UNKNOWN'], fill: colors.reasoning },
    { id: 'method', x: 70, y: 950, width: 360, height: 75, lines: ['方法证据组', '实施了载体相关实验或观测'], fill: colors.reasoning },
    { id: 'carrier', x: 480, y: 950, width: 360, height: 75, lines: ['载体证据组', '识别具体载磁矿物或载体'], fill: colors.reasoning },
    { id: 'bridge', x: 890, y: 950, width: 360, height: 75, lines: ['意义桥接证据组', '载体关联分量、原生/次生解释'], fill: colors.reasoning },
    { id: 'conflict', x: 1300, y: 950, width: 420, height: 75, lines: ['显式反向证据', '论文明确报告未能确定载体'], fill: colors.conflict, stroke: colors.conflictBorder },
    { id: 'chain', x: 300, y: 1080, width: 420, height: 75, lines: ['连接证据链检验', '同一目标 + 页邻近 + 作用域兼容'], fill: colors.reasoning },
    { id: 'engine', x: 760, y: 1080, width: 420, height: 75, lines: ['专家知识冲突聚合', '规则去重 / 限定例外 / 保留矛盾'], fill: colors.reasoning },
    { id: 'state', x: 1260, y: 1080, width: 420, height: 75, lines: ['四值证据状态', 'SUPPORTED_ONLY / REFUTED_ONLY', 'BOTH / NEITHER'], fill: colors.neutral },

    { id: 'normative', x: 210, y: 1260, width: 430, height: 90, lines: ['规范科学判决', '必需组齐全：1；明确冲突：0', '其余情况：ABSTAIN'], fill: colors.output, stroke: colors.outputBorder },
    { id: 'adapter', x: 700, y: 1260, width: 380, height: 90, lines: ['业务二元兼容输出', 'UNKNOWN 映射为 0', '同时保留人工复核标志'], fill: colors.neutral },
    { id: 'output', x: 1140, y: 1260, width: 470, height: 90, lines: ['R3 可追溯证据包', '判决 / 缺失组 / 原文引文 / 页码', 'scope_id / 理由 / 输入与知识哈希'], fill: colors.output, stroke: colors.outputBorder },
  ];

  const edges = [
    { points: [[315, 190], [315, 280]], arrow: false },
    { points: [[900, 190], [900, 280]], arrow: false },
    { points: [[205, 280], [900, 280]], arrow: false },
    { points: [[205, 280], [205, 300]] },
    { points: [[330, 340], [370, 340]] },
    { points: [[650, 340], [700, 340]] },
    { points: [[900, 340], [1030, 340]], label: '无风险', labelX: 965, labelY: 326, labelWidth: 68 },
    { points: [[800, 395], [800, 415], [970, 415], [970, 430]], label: '有风险', labelX: 875, labelY: 405, labelWidth: 68 },
    { points: [[1120, 465], [1180, 465], [1180, 380]] },
    { points: [[1330, 340], [1410, 340]] },

    { points: [[100, 158], [25, 158], [25, 660], [70, 660]] },
    { points: [[1560, 380], [1560, 565]], arrow: false },
    { points: [[1485, 190], [1740, 190], [1740, 565], [1560, 565]], arrow: false, dashed: true },
    { points: [[545, 565], [1560, 565]], arrow: false },
    { points: [[545, 565], [545, 620]] },
    { points: [[360, 660], [400, 660]] },
    { points: [[690, 660], [730, 660]] },
    { points: [[1020, 660], [1060, 660]] },
    { points: [[1360, 660], [1400, 660]] },
    { points: [[1550, 700], [1550, 770], [900, 770], [900, 830]] },

    { points: [[900, 900], [900, 930]], arrow: false },
    { points: [[250, 930], [1510, 930]], arrow: false },
    { points: [[250, 930], [250, 950]] },
    { points: [[660, 930], [660, 950]] },
    { points: [[1070, 930], [1070, 950]] },
    { points: [[1510, 930], [1510, 950]] },
    { points: [[250, 1025], [250, 1050]], arrow: false },
    { points: [[660, 1025], [660, 1050]], arrow: false },
    { points: [[1070, 1025], [1070, 1050]], arrow: false },
    { points: [[250, 1050], [1070, 1050]], arrow: false },
    { points: [[510, 1050], [510, 1080]] },
    { points: [[1510, 1025], [1510, 1063], [970, 1063], [970, 1080]] },
    { points: [[720, 1118], [760, 1118]] },
    { points: [[1180, 1118], [1260, 1118]] },

    { points: [[1470, 1155], [1470, 1190], [425, 1190], [425, 1260]] },
    { points: [[640, 1305], [700, 1305]] },
    { points: [[1080, 1305], [1140, 1305]] },
  ];

  return {
    width: 1800,
    height: 1410,
    title: '图 1  R3 原文取证与规范推理流程',
    description: 'A fixed-grid scientific workflow for R3 source acquisition, parsing, target binding, evidence-chain reasoning, normative decision, and auditable output.',
    bands: [
      { x: 40, y: 80, width: 1720, height: 130, label: 'A  输入与数据边界' },
      { x: 40, y: 230, width: 1720, height: 300, label: 'B  原文解析与页级质量控制' },
      { x: 40, y: 550, width: 1720, height: 210, label: 'C  领域检索、作用域与目标归属' },
      { x: 40, y: 780, width: 1720, height: 400, label: 'D  R3 证据链与冲突推理' },
      { x: 40, y: 1200, width: 1720, height: 180, label: 'E  规范判决与可追溯输出' },
    ],
    nodes,
    edges,
    junctions: [
      [205, 280], [315, 280], [900, 280],
      [545, 565], [1560, 565],
      [250, 930], [660, 930], [900, 930], [1070, 930], [1510, 930],
      [250, 1050], [510, 1050], [660, 1050], [1070, 1050],
    ],
  };
}

function supervisedEvaluationFigure() {
  const nodes = [
    { id: 'cache', x: 70, y: 125, width: 370, height: 65, lines: ['融合页级原文缓存', '主缓存优先；MinerU 仅补齐空文献'], fill: colors.cache },
    { id: 'knowledge', x: 500, y: 125, width: 370, height: 65, lines: ['冻结聚合专家知识', '术语图谱 / 诊断与载体模式'], fill: colors.knowledge },
    { id: 'identity', x: 930, y: 125, width: 370, height: 65, lines: ['目标身份清单', 'REFNO / RESULTNO / ROCKNAME'], fill: colors.input },
    { id: 'labels', x: 1360, y: 125, width: 370, height: 65, lines: ['历史 R3 标签', '特征与折冻结后才可读取'], fill: colors.label, stroke: colors.labelBorder },

    { id: 'style', x: 70, y: 300, width: 430, height: 75, lines: ['文献结构与来源质量特征', '年代 / 章节 / 图表 / 解析质量'], fill: colors.binding },
    { id: 'article', x: 685, y: 300, width: 430, height: 75, lines: ['全文专家概念特征', '方法 / 诊断 / 载体的频次与多样性'], fill: colors.retrieval },
    { id: 'local', x: 1300, y: 300, width: 430, height: 75, lines: ['目标邻域与身份特征', '目标提及 / 局部概念 / 目标数量'], fill: colors.retrieval },
    { id: 'freeze', x: 220, y: 440, width: 420, height: 75, lines: ['无标签特征矩阵冻结', '固定列顺序 / 行对齐 / 内容哈希'], fill: colors.reasoning },
    { id: 'folds', x: 700, y: 440, width: 420, height: 75, lines: ['确定性 REFNO 分组五折', '同一论文的全部目标只进入一折'], fill: colors.reasoning },
    { id: 'receipt', x: 1180, y: 440, width: 420, height: 75, lines: ['候选配置冻结回执', '特征 / 折 / 四模型 / 运行环境'], fill: colors.neutral },

    { id: 'train', x: 650, y: 620, width: 500, height: 75, lines: ['训练折标签打开', '测试 REFNO 完全排除'], fill: colors.label, stroke: colors.labelBorder },
    { id: 'weight', x: 650, y: 730, width: 500, height: 70, lines: ['训练样本权重校正', '降低同论文同目标重复行的支配'], fill: colors.knowledge },
    { id: 'hgb', x: 70, y: 850, width: 360, height: 75, lines: ['直方图梯度提升树', '学习非线性阈值与特征交互'], fill: colors.reasoning },
    { id: 'rf', x: 500, y: 850, width: 360, height: 75, lines: ['随机森林', '对训练样本与特征进行树集成'], fill: colors.reasoning },
    { id: 'extra', x: 940, y: 850, width: 360, height: 75, lines: ['极端随机树', '增加候选切分随机性'], fill: colors.reasoning },
    { id: 'logreg', x: 1380, y: 850, width: 360, height: 75, lines: ['标准化逻辑回归', '提供线性概率补充'], fill: colors.reasoning },
    { id: 'oof', x: 650, y: 980, width: 500, height: 70, lines: ['四模型外折概率', '被预测论文未参与对应模型拟合'], fill: colors.reasoning },

    { id: 'ensemble', x: 100, y: 1120, width: 400, height: 80, lines: ['四模型概率等权平均', '形成预先冻结的 mean4 主概率'], fill: colors.reasoning },
    { id: 'practice', x: 700, y: 1120, width: 400, height: 80, lines: ['历史 R3 标签兼容概率', '每个目标保留概率与折归属'], fill: colors.knowledge },
    { id: 'selective', x: 1300, y: 1120, width: 400, height: 80, lines: ['单目标选择性判定', '高置信输出 0/1', '中间概率区间 ABSTAIN'], fill: colors.conflict, stroke: colors.conflictBorder },
    { id: 'oofReceipt', x: 100, y: 1260, width: 400, height: 80, lines: ['全覆盖 OOF 评测', '模型指标 / 混淆矩阵 / 逐折结果'], fill: colors.output, stroke: colors.outputBorder },
    { id: 'errors', x: 700, y: 1260, width: 400, height: 80, lines: ['分组误差分析', '单目标 / 多目标 / 同文献混合标签'], fill: colors.neutral },
    { id: 'selectiveReceipt', x: 1300, y: 1260, width: 400, height: 80, lines: ['选择性评测回执', '已判定准确率与覆盖率成对报告', '仅适用于单目标开发子集'], fill: colors.output, stroke: colors.outputBorder },
    { id: 'boundary', x: 700, y: 1385, width: 400, height: 70, lines: ['评测结论边界', '内部 REFNO 分组 OOF', '不等同于独立外部验证'], fill: colors.label, stroke: colors.labelBorder },
  ];

  const edges = [
    { points: [[255, 190], [255, 280]], arrow: false },
    { points: [[685, 190], [685, 280]], arrow: false },
    { points: [[1115, 190], [1115, 280]], arrow: false },
    { points: [[255, 280], [1515, 280]], arrow: false },
    { points: [[285, 280], [285, 300]] },
    { points: [[900, 280], [900, 300]] },
    { points: [[1515, 280], [1515, 300]] },

    { points: [[285, 375], [285, 405]], arrow: false },
    { points: [[900, 375], [900, 405]], arrow: false },
    { points: [[1515, 375], [1515, 405]], arrow: false },
    { points: [[285, 405], [1515, 405]], arrow: false },
    { points: [[430, 405], [430, 440]] },
    { points: [[640, 478], [700, 478]] },
    { points: [[1120, 478], [1180, 478]] },

    { points: [[910, 515], [910, 620]] },
    { points: [[1545, 190], [1775, 190], [1775, 658], [1150, 658]], dashed: true },
    { points: [[900, 695], [900, 730]] },
    { points: [[900, 800], [900, 825]], arrow: false },
    { points: [[250, 825], [1560, 825]], arrow: false },
    { points: [[250, 825], [250, 850]] },
    { points: [[680, 825], [680, 850]] },
    { points: [[1120, 825], [1120, 850]] },
    { points: [[1560, 825], [1560, 850]] },
    { points: [[250, 925], [250, 950]], arrow: false },
    { points: [[680, 925], [680, 950]], arrow: false },
    { points: [[1120, 925], [1120, 950]], arrow: false },
    { points: [[1560, 925], [1560, 950]], arrow: false },
    { points: [[250, 950], [1560, 950]], arrow: false },
    { points: [[900, 950], [900, 980]] },

    { points: [[900, 1050], [900, 1080], [25, 1080], [25, 1160], [100, 1160]] },
    { points: [[500, 1160], [700, 1160]] },
    { points: [[1100, 1160], [1300, 1160]] },
    { points: [[900, 1200], [900, 1230], [300, 1230], [300, 1260]] },
    { points: [[500, 1300], [700, 1300]] },
    { points: [[1500, 1200], [1500, 1260]] },
    { points: [[900, 1340], [900, 1365]], arrow: false },
    { points: [[1500, 1340], [1500, 1365]], arrow: false },
    { points: [[900, 1365], [1500, 1365]], arrow: false },
    { points: [[900, 1365], [900, 1385]] },
  ];

  return {
    width: 1800,
    height: 1500,
    title: '图 2  R3 历史实践监督评测流程',
    description: 'A fixed-grid scientific workflow for label-free feature freezing, REFNO-grouped out-of-fold training, ensemble prediction, selective abstention, error analysis, and external validation.',
    bands: [
      { x: 40, y: 80, width: 1720, height: 130, label: 'A  输入边界：标签与运行时原文分离' },
      { x: 40, y: 230, width: 1720, height: 310, label: 'B  标签打开前的特征、分组折与候选冻结' },
      { x: 40, y: 560, width: 1720, height: 510, label: 'C  外折训练与论文隔离预测' },
      { x: 40, y: 1090, width: 1720, height: 390, label: 'D  概率集成、评测与结论边界' },
    ],
    nodes,
    edges,
    junctions: [
      [255, 280], [285, 280], [685, 280], [900, 280], [1115, 280], [1515, 280],
      [285, 405], [430, 405], [900, 405], [1515, 405],
      [250, 825], [680, 825], [900, 825], [1120, 825], [1560, 825],
      [250, 950], [680, 950], [900, 950], [1120, 950], [1560, 950],
      [900, 1365], [1500, 1365],
    ],
  };
}

async function writeFigure(name, spec) {
  validateFigure(spec);
  const svgPath = path.join(figureDirectory, `${name}.svg`);
  const pngPath = path.join(figureDirectory, `${name}.png`);
  const svg = figureSvg(spec);
  fs.writeFileSync(svgPath, svg, 'utf8');
  await sharp(Buffer.from(svg), { density: 240 })
    .flatten({ background: colors.background })
    .resize({ width: 4800, withoutEnlargement: true })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(pngPath);
  const metadata = await sharp(pngPath).metadata();
  process.stdout.write(`${name}: ${metadata.width}x${metadata.height}\n`);
}

async function main() {
  fs.mkdirSync(figureDirectory, { recursive: true });
  await writeFigure('r3-original-evidence-normative-workflow', originalEvidenceFigure());
  await writeFigure('r3-supervised-evaluation-workflow', supervisedEvaluationFigure());
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
