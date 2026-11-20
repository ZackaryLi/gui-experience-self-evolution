VISUAL_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Trajectory & Intent Evaluation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <style>
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
        .timeline-line::before {
            content: ''; position: absolute; left: 15px; top: 30px; bottom: -30px;
            width: 2px; background-color: #e5e7eb; z-index: 0;
        }
        .step-node:last-child .timeline-line::before { display: none; }
    </style>
</head>
<body class="bg-gray-50 text-gray-800 font-sans h-screen flex overflow-hidden">

<div id="app" class="flex w-full h-full">

    <!-- 左侧：Intent 列表 -->
    <aside class="w-72 bg-white border-r shadow-sm flex flex-col z-10 flex-shrink-0">
        <div class="p-4 border-b bg-gray-900 text-white">
            <h1 class="text-lg font-bold">Agent 评估控制台</h1>
            <p class="text-xs text-gray-400 mt-1">Paired Trajectory Analysis</p>
        </div>
        <div class="flex-1 overflow-y-auto p-3">
            <div v-if="loadingIntents" class="text-sm text-gray-500 animate-pulse">加载意图列表中...</div>
            <ul v-else class="space-y-1">
                <li v-for="intent in intentList" :key="intent"
                    @click="selectIntent(intent)"
                    :class="['px-3 py-2 rounded cursor-pointer text-sm font-medium transition-colors duration-150',
                             currentIntent === intent ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-sm' : 'hover:bg-gray-100 text-gray-600']">
                    {{ intent.replace('.json', '') }}
                </li>
            </ul>
        </div>
    </aside>

    <!-- 右侧：主体内容区 -->
    <main class="flex-1 flex flex-col h-full overflow-hidden bg-gray-50">

        <div v-if="!currentIntent" class="flex-1 flex items-center justify-center text-gray-400 text-lg">
            👈 请在左侧选择一个用户任务意图进行分析
        </div>

        <template v-else>
            <!-- 顶部：意图经验 (Generated Plan) 展示 - 【已优化：压缩高度和间距】 -->
            <section class="p-3 border-b bg-white flex-shrink-0 max-h-[28%] overflow-y-auto shadow-sm z-10">
                <div class="flex justify-between items-center mb-2">
                    <div class="flex items-center gap-3">
                        <h2 class="text-lg font-bold text-gray-800">{{ currentIntentName }}</h2>
                        <!-- 统计信息：全局平均步数 -->
                        <div class="flex bg-gray-100 rounded-lg p-0.5 px-2 text-xs border border-gray-200 shadow-inner">
                            <div class="border-r border-gray-300 pr-2 mr-2">
                                <span class="text-gray-500">无经验均步:</span>
                                <span class="font-bold text-gray-800">{{ avgNoPlanSteps }}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">带经验均步:</span>
                                <span class="font-bold text-blue-600">{{ avgWithPlanSteps }}</span>
                            </div>
                        </div>
                    </div>

                    <!-- 轨迹详情开关 -->
                    <button @click="showDetailsMode = !showDetailsMode"
                            :class="['px-3 py-1 rounded-full text-xs font-bold border transition-colors',
                                    showDetailsMode ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50']">
                        {{ showDetailsMode ? '👁️ 图文详情 (开)' : '⚡ 精简动作 (开)' }}
                    </button>
                </div>

                <div v-if="intentData && intentData.generated_plans" class="space-y-2">
                    <p class="text-xs text-gray-600 bg-gray-50 p-1.5 rounded border border-gray-200 truncate" :title="intentData.generated_plans.summary">
                        <span class="font-bold text-gray-800">方案总结：</span>{{ intentData.generated_plans.summary }}
                    </p>

                    <div class="flex gap-2 overflow-x-auto pb-1">
                        <!-- 经验步骤卡片 - 【已优化：减小卡片宽度和内边距】 -->
                        <div v-for="phase in intentData.generated_plans.execution_plan" :key="phase.phase_idx"
                             class="flex-shrink-0 w-72 bg-white border rounded-lg p-2 shadow-sm flex flex-col">
                            <div class="flex items-center gap-1.5 mb-1.5">
                                <span class="bg-blue-600 text-white text-[10px] font-bold w-4 h-4 flex items-center justify-center rounded-full">{{ phase.phase_idx }}</span>
                                <h3 class="font-bold text-xs text-gray-800 truncate" :title="phase.phase_goal">{{ phase.phase_goal }}</h3>
                            </div>

                            <!-- 仅展示动作 -->
                            <ul class="text-[11px] text-gray-600 list-disc list-inside mb-1.5 space-y-0.5">
                                <li v-for="(action, idx) in phase.mapped_actions" :key="idx" class="truncate" :title="action">{{ action }}</li>
                            </ul>

                            <!-- 下拉详情按钮及内容 -->
                            <div class="mt-auto pt-1.5 border-t border-gray-100">
                                <button @click="phase._expanded = !phase._expanded" class="text-[11px] text-blue-600 hover:text-blue-800 w-full text-left font-medium">
                                    {{ phase._expanded ? '收起详情 ▲' : '查看详情 ▼' }}
                                </button>
                                <div v-show="phase._expanded" class="mt-1.5 space-y-1.5">
                                    <div v-if="phase.expected_visual_state" class="text-[11px] bg-gray-100 p-1.5 rounded text-gray-600 leading-snug">
                                        <span class="font-bold">👁️ 预期：</span>{{ phase.expected_visual_state }}
                                    </div>
                                    <div v-if="phase.attention_warning" class="text-[11px] bg-red-50 border border-red-200 text-red-600 p-1.5 rounded font-medium leading-snug">
                                        <span class="font-bold">⚠️ 防错：</span>{{ phase.attention_warning }}
                                    </div>
                                    <div v-if="!phase.expected_visual_state && !phase.attention_warning" class="text-[11px] text-gray-400 italic">
                                        无额外详情
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- 底部：成对轨迹展示区 (按 Query 同步滑动) -->
            <section class="flex-1 flex flex-col overflow-hidden">

                <!-- 统一的 Query 导航栏 -->
                <div class="bg-slate-800 text-white px-5 py-2 flex justify-between items-center shadow-md z-20 flex-shrink-0">
                    <div class="flex items-center gap-4 flex-1 overflow-hidden mr-4">
                        <span class="font-bold whitespace-nowrap text-sm text-slate-300">当前任务指令:</span>
                        <div class="text-sm font-medium bg-slate-700 px-4 py-1 rounded-full truncate flex-1"
                             :title="currentPair?.query">
                            {{ loadingTrajectories ? '正在匹配数据...' : (currentPair?.query || '未找到对应的任务指令') }}
                        </div>
                    </div>

                    <div v-if="pairedTrajectories.length > 0" class="flex items-center gap-4 flex-shrink-0">
                        <span class="text-sm font-medium text-slate-300">
                            指令 <span class="text-white font-bold">{{ currentPairIndex + 1 }}</span> / {{ pairedTrajectories.length }}
                        </span>
                        <div class="flex gap-1">
                            <button @click="currentPairIndex > 0 && currentPairIndex--" :disabled="currentPairIndex === 0" class="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-bold">◀</button>
                            <button @click="currentPairIndex < pairedTrajectories.length - 1 && currentPairIndex++" :disabled="currentPairIndex === pairedTrajectories.length - 1" class="px-2 py-0.5 rounded bg-slate-600 hover:bg-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-bold">▶</button>
                        </div>
                    </div>
                </div>

                <!-- 双轨对比面板 -->
                <div class="flex-1 flex overflow-hidden">

                    <!-- 左半屏：无 Plan -->
                    <div class="flex-1 border-r bg-gray-50 flex flex-col h-full overflow-hidden">
                        <div class="bg-gray-200 px-4 py-1.5 text-sm font-bold text-gray-700 flex justify-between items-center shadow-sm z-10 flex-shrink-0">
                            <span>无经验 (No Plan) 轨迹</span>
                            <span v-if="noPlanTrajectory?.records" class="bg-gray-300 px-2 py-0.5 rounded text-gray-800 text-xs">总计 {{ noPlanTrajectory.stepCount }} 步</span>
                        </div>

                        <div class="flex-1 overflow-y-auto p-3 relative">
                            <div v-if="loadingTrajectories" class="text-sm text-gray-500">正在解析执行记录...</div>
                            <div v-else-if="!noPlanTrajectory?.records" class="text-sm text-red-500 flex flex-col items-center justify-center h-32 bg-white rounded border border-dashed">
                                <span>当前 Query 未找到【无经验】轨迹数据</span>
                            </div>
                            <div v-else class="space-y-4">
                                <trajectory-node v-for="step in noPlanTrajectory.records" :key="step.step_id" :step="step" :base-url="noPlanTrajectory.baseUrl" :show-details="showDetailsMode"></trajectory-node>
                            </div>
                        </div>
                    </div>

                    <!-- 右半屏：有 Plan -->
                    <div class="flex-1 bg-white flex flex-col h-full overflow-hidden">
                        <div class="bg-blue-50 border-b border-blue-100 px-4 py-1.5 text-sm font-bold text-blue-800 flex justify-between items-center shadow-sm z-10 flex-shrink-0">
                            <span>带经验 (Generated Plan) 轨迹</span>
                            <span v-if="withPlanTrajectory?.records" class="bg-blue-200 px-2 py-0.5 rounded text-blue-900 text-xs">总计 {{ withPlanTrajectory.stepCount }} 步</span>
                        </div>

                        <div class="flex-1 overflow-y-auto p-3 relative">
                            <div v-if="loadingTrajectories" class="text-sm text-gray-500">正在解析执行记录...</div>
                            <div v-else-if="!withPlanTrajectory?.records" class="text-sm text-red-500 flex flex-col items-center justify-center h-32 bg-white rounded border border-dashed">
                                <span>当前 Query 未找到【带经验】轨迹数据</span>
                            </div>
                            <div v-else class="space-y-4">
                                <trajectory-node v-for="step in withPlanTrajectory.records" :key="step.step_id" :step="step" :base-url="withPlanTrajectory.baseUrl" :show-details="showDetailsMode"></trajectory-node>
                            </div>
                        </div>
                    </div>
                </div>

            </section>
        </template>
    </main>
</div>

<!-- Vue 组件模板保持不变 -->
<template id="trajectory-node-template">
    <div class="step-node relative pl-10">
        <div class="timeline-line"></div>
        <div class="absolute left-0 top-0 w-8 h-8 rounded-full bg-white border-2 flex items-center justify-center text-xs font-bold z-10 shadow-sm"
             :class="showDetails ? 'border-gray-300 text-gray-500' : 'border-indigo-300 text-indigo-600'">
            {{ step.step_id }}
        </div>

        <div class="bg-white border rounded-lg shadow-sm overflow-hidden flex flex-col md:flex-row transition-all hover:shadow-md">
            <!-- 截图区域 (仅在详情模式展示) -->
            <div v-if="showDetails" class="w-full md:w-1/3 bg-gray-100 border-r flex items-center justify-center p-2 flex-shrink-0">
                <img v-if="step.screenshot" :src="`${baseUrl}/${step.screenshot}`" class="max-h-56 object-contain rounded border border-gray-200 shadow-sm" alt="Screenshot" @error="handleImgError">
                <div v-else class="text-gray-400 text-xs">无截图</div>
            </div>

            <!-- 信息区域 (精简/详情自适应) -->
            <div class="p-3 flex flex-col text-sm w-full">
                <!-- 核心：Action与耗时 -->
                <div class="flex justify-between items-start mb-2">
                    <div class="flex flex-wrap items-center gap-2">
                        <span v-if="step.model_output?.output?.action" class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-mono font-bold shadow-sm">
                            {{ step.model_output.output.action }}
                        </span>
                        <span class="inline-block bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs border">
                            ⏱️ {{ step.elapsed_time ? step.elapsed_time.toFixed(2) : 'N/A' }}s
                        </span>
                    </div>
                </div>

                <!-- 动作目的 (无论模式均展示) -->
                <div :class="['mb-2', showDetails ? '' : 'mb-0']">
                    <span class="font-bold text-gray-700 text-xs">🎯 动作目的: </span>
                    <span class="text-gray-600 text-xs">{{ step.model_output?.output?.action_purpose || '无' }}</span>
                </div>

                <!-- 详情模式下展示的信息 -->
                <template v-if="showDetails">
                    <!-- 思考过程 -->
                    <div class="mb-3 mt-2">
                        <p class="font-bold text-gray-700 text-xs mb-1">🧠 Thought (思考过程):</p>
                        <p class="text-gray-600 bg-yellow-50 p-2 rounded text-xs border border-yellow-100 whitespace-pre-line leading-relaxed">{{ step.model_output?.output?.thought || '无' }}</p>
                    </div>

                    <!-- 折叠的底层数据 -->
                    <div class="mt-auto pt-2 border-t">
                        <button @click="showActionSeq = !showActionSeq" class="text-xs text-gray-500 hover:text-gray-800 flex items-center gap-1 focus:outline-none font-medium">
                            <span>{{ showActionSeq ? '▼ 收起底层数据' : '▶ 展开底层 JSON 数据' }}</span>
                        </button>
                        <div v-show="showActionSeq" class="mt-2 bg-gray-900 text-green-400 p-2 rounded overflow-x-auto text-xs font-mono">
                            <pre><code>{{ JSON.stringify(step.action_seq, null, 2) }}</code></pre>
                        </div>
                    </div>
                </template>
            </div>
        </div>
    </div>
</template>

<script>
const { createApp, ref, computed, reactive } = Vue;

const TrajectoryNode = {
    template: '#trajectory-node-template',
    props: ['step', 'baseUrl', 'showDetails'],
    setup() {
        const showActionSeq = ref(false);
        const handleImgError = (e) => {
            e.target.style.display = 'none';
            e.target.insertAdjacentHTML('afterend', '<div class="text-gray-400 text-xs text-center w-full">图片加载失败或丢失</div>');
        };
        return { showActionSeq, handleImgError };
    }
};

createApp({
    components: { TrajectoryNode },
    setup() {
        const intentList = ref([]);
        const loadingIntents = ref(true);
        const currentIntent = ref(null);

        const intentData = ref(null);
        const showDetailsMode = ref(true); // 全局详情开关

        // 核心配对数据：按 query 组织的数组
        // 数据结构: [ { query: "...", noPlan: { ... }, withPlan: { ... } }, ... ]
        const pairedTrajectories = ref([]);
        const currentPairIndex = ref(0);
        const loadingTrajectories = ref(false);

        const currentIntentName = computed(() => currentIntent.value ? currentIntent.value.replace('.json', '') : '');

        // 提取当前视图对应的数据
        const currentPair = computed(() => pairedTrajectories.value[currentPairIndex.value] || null);
        const noPlanTrajectory = computed(() => currentPair.value?.noPlan || null);
        const withPlanTrajectory = computed(() => currentPair.value?.withPlan || null);

        // 计算全局平均步骤数
        const avgNoPlanSteps = computed(() => {
            const valid = pairedTrajectories.value.filter(p => p.noPlan);
            if (!valid.length) return '0.0';
            const sum = valid.reduce((acc, p) => acc + p.noPlan.stepCount, 0);
            return (sum / valid.length).toFixed(1);
        });

        const avgWithPlanSteps = computed(() => {
            const valid = pairedTrajectories.value.filter(p => p.withPlan);
            if (!valid.length) return '0.0';
            const sum = valid.reduce((acc, p) => acc + p.withPlan.stepCount, 0);
            return (sum / valid.length).toFixed(1);
        });

        const init = async () => {
            try {
                const res = await fetch('./intents/intents_index.json');
                const data = await res.json();
                intentList.value = data.intents || [];
            } catch (err) {
                console.error("加载配置失败", err);
            } finally {
                loadingIntents.value = false;
            }
        };

        const selectIntent = async (intentFile) => {
            if (currentIntent.value === intentFile) return;

            currentIntent.value = intentFile;
            intentData.value = null;
            pairedTrajectories.value = [];
            currentPairIndex.value = 0;
            const intentName = currentIntentName.value;

            // 加载经验JSON
            try {
                const res = await fetch(`./intents/${intentFile}`);
                const data = await res.json();
                if(data.generated_plans && data.generated_plans.execution_plan) {
                    data.generated_plans.execution_plan.forEach(phase => phase._expanded = false);
                }
                intentData.value = reactive(data);
            } catch (err) {
                console.error(`加载经验文件失败`, err);
            }

            // 同步加载并匹配两边的轨迹
            await loadAndPairTrajectories(intentName);
        };

        // 获取单个目录下的所有轨迹数据
        const fetchTrajectories = async (intentName, taskType) => {
            const results = [];
            const dataDirPath = `./evaluation/${intentName}/${taskType}/data/`;
            try {
                const dirRes = await fetch(dataDirPath);
                if (!dirRes.ok) return results;

                const htmlText = await dirRes.text();
                const doc = new DOMParser().parseFromString(htmlText, 'text/html');

                const folders = Array.from(doc.querySelectorAll('a'))
                    .map(a => a.getAttribute('href'))
                    .filter(href => href && href.endsWith('/') && href !== '../' && !href.startsWith('/'));

                for (const folder of folders) {
                    const targetBaseUrl = `${dataDirPath}${folder}`.replace(/\/$/, '');
                    try {
                        const recordRes = await fetch(`${targetBaseUrl}/record.json`);
                        if (recordRes.ok) {
                            const rawRecord = await recordRes.json();
                            const records = Array.isArray(rawRecord) ? rawRecord : (rawRecord.data || rawRecord[0] || []);

                            if (records.length > 0) {
                                results.push({
                                    folderName: folder,
                                    baseUrl: targetBaseUrl,
                                    records: records,
                                    query: records[0].query || '未获取到Query',
                                    stepCount: records.length
                                });
                            }
                        }
                    } catch (e) {
                        console.warn(`跳过损坏的记录 ${targetBaseUrl}`, e);
                    }
                }
            } catch (err) {
                console.warn(`扫描 ${taskType} 轨迹失败:`, err);
            }
            return results;
        };

        // 核心：加载并按 Query 配对
        const loadAndPairTrajectories = async (intentName) => {
            loadingTrajectories.value = true;

            // 并行加载两侧数据
            const [noPlanList, withPlanList] = await Promise.all([
                fetchTrajectories(intentName, 'task_no_plan'),
                fetchTrajectories(intentName, 'task_generated_plan')
            ]);

            const queryMap = new Map();

            // 辅助函数：合并到 Map 中
            const addToList = (list, type) => {
                list.forEach(item => {
                    const q = item.query;
                    if (!queryMap.has(q)) {
                        queryMap.set(q, { query: q, noPlan: null, withPlan: null });
                    }
                    // 如果存在多条相同 query，默认展示抓取到的第一条
                    if (!queryMap.get(q)[type]) {
                        queryMap.get(q)[type] = item;
                    }
                });
            };

            addToList(noPlanList, 'noPlan');
            addToList(withPlanList, 'withPlan');

            // 转换成数组供 Vue 渲染
            pairedTrajectories.value = Array.from(queryMap.values());
            loadingTrajectories.value = false;
        };

        init();

        return {
            intentList, loadingIntents, currentIntent, currentIntentName, selectIntent,
            intentData, showDetailsMode,
            pairedTrajectories, currentPairIndex, currentPair,
            noPlanTrajectory, withPlanTrajectory,
            avgNoPlanSteps, avgWithPlanSteps,
            loadingTrajectories
        };
    }
}).mount('#app');
</script>
</body>
</html>
"""