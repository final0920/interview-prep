<template>
  <div class="view-shell">
    <div class="view-header">
      <h1 class="view-title">复习计划</h1>
      <button class="btn btn-secondary" @click="reviewStore.fetch()">刷新</button>
    </div>

    <div v-if="reviewStore.loading" class="status-msg">加载中...</div>
    <div v-else-if="reviewStore.error" class="status-msg error">{{ reviewStore.error }}</div>

    <div v-else class="review-layout">
      <!-- Quality report -->
      <section class="quality-section">
        <div class="section-title">质量报告</div>
        <div v-if="!reviewStore.qualityReport" class="empty-msg">暂无报告数据</div>
        <template v-else>
          <!-- Key metrics -->
          <div class="metrics-row">
            <div class="metric-card">
              <div class="metric-val">{{ pct(reviewStore.qualityReport.pass_rate) }}</div>
              <div class="metric-label">通过率</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{{ pct(reviewStore.qualityReport.grounding_rate) }}</div>
              <div class="metric-label">引用率</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{{ reviewStore.qualityReport.total_questions ?? '-' }}</div>
              <div class="metric-label">题目总数</div>
            </div>
          </div>

          <!-- Type distribution -->
          <div v-if="reviewStore.qualityReport.type_dist" class="subsection">
            <div class="subsection-title">题型分布</div>
            <div class="type-dist">
              <div
                v-for="(count, qtype) in reviewStore.qualityReport.type_dist"
                :key="qtype"
                class="type-row"
              >
                <span class="type-label">{{ qtype }}</span>
                <div class="type-bar-wrap">
                  <div
                    class="type-bar"
                    :style="{ width: barWidth(count, reviewStore.qualityReport.total_questions) }"
                  />
                </div>
                <span class="type-count">{{ count }}</span>
              </div>
            </div>
          </div>

          <!-- Redlines -->
          <div v-if="reviewStore.qualityReport.redlines?.length" class="redlines-section">
            <div class="subsection-title">红线警告</div>
            <div v-for="rl in reviewStore.qualityReport.redlines" :key="rl" class="redline-item">
              {{ rl }}
            </div>
          </div>
        </template>
      </section>

      <!-- SM-2 schedule table -->
      <section class="schedule-section">
        <div class="section-title">复习时间表 (SM-2)</div>
        <div v-if="!reviewStore.schedule.length" class="empty-msg">暂无复习条目</div>
        <div v-else class="schedule-table-wrap">
          <table class="schedule-table">
            <thead>
              <tr>
                <th>题目</th>
                <th>下次复习</th>
                <th>间隔(天)</th>
                <th>EF</th>
                <th>次数</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="item in sortedSchedule"
                :key="item.question_id ?? item.id"
                :class="{ 'row-due': isDue(item.due_ts) }"
              >
                <td class="td-question">{{ item.question_text ?? item.question_id ?? '-' }}</td>
                <td>{{ formatDue(item.due_ts) }}</td>
                <td>{{ item.interval ?? '-' }}</td>
                <td>{{ item.ef != null ? item.ef.toFixed(2) : '-' }}</td>
                <td>{{ item.reps ?? '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useReviewStore } from '../stores/review'

const reviewStore = useReviewStore()

onMounted(() => reviewStore.fetch())

const sortedSchedule = computed(() =>
  [...reviewStore.schedule].sort((a, b) => (a.due_ts ?? 0) - (b.due_ts ?? 0))
)

function pct(val) {
  if (val == null) return '-'
  return `${Math.round(val * 100)}%`
}

function barWidth(count, total) {
  if (!total) return '0%'
  return `${Math.round((count / total) * 100)}%`
}

function formatDue(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleDateString('zh-CN')
}

function isDue(ts) {
  if (!ts) return false
  return ts * 1000 <= Date.now()
}
</script>

<style scoped>
.view-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  border-bottom: 1px solid #2d3148;
  background: #1a1d27;
  flex-shrink: 0;
}

.view-title {
  font-size: 16px;
  font-weight: 600;
  color: #e2e8f0;
}

.btn { padding: 6px 14px; border: none; border-radius: 6px; font-size: 13px; cursor: pointer; }
.btn-secondary { background: #2d3148; color: #94a3b8; }
.btn-secondary:hover { background: #3d4160; }

.status-msg { padding: 24px; font-size: 14px; color: #4b5563; }
.status-msg.error { color: #f87171; }

.review-layout {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  color: #7c86ff;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 14px;
}

.empty-msg { font-size: 13px; color: #4b5563; }

/* Quality section */
.quality-section {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 10px;
  padding: 16px;
}

.metrics-row {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.metric-card {
  flex: 1;
  background: #13151f;
  border: 1px solid #2d3148;
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}

.metric-card .metric-val {
  font-size: 22px;
  font-weight: 700;
  color: #7c86ff;
}

.metric-card .metric-label {
  font-size: 11px;
  color: #6b7280;
  margin-top: 4px;
  text-transform: uppercase;
}

.subsection { margin-top: 14px; }
.subsection-title {
  font-size: 11px;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

/* Type dist bars */
.type-dist { display: flex; flex-direction: column; gap: 6px; }
.type-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.type-label { width: 100px; color: #94a3b8; flex-shrink: 0; }
.type-bar-wrap {
  flex: 1;
  height: 4px;
  background: #2d3148;
  border-radius: 2px;
  overflow: hidden;
}
.type-bar {
  height: 100%;
  background: #7c86ff;
  border-radius: 2px;
  transition: width 0.4s ease;
}
.type-count { width: 24px; text-align: right; color: #e2e8f0; }

/* Redlines */
.redlines-section { margin-top: 14px; }
.redline-item {
  padding: 8px 12px;
  background: rgba(239,68,68,0.1);
  border: 1px solid rgba(239,68,68,0.25);
  border-radius: 6px;
  font-size: 13px;
  color: #fca5a5;
  margin-bottom: 6px;
}

/* Schedule table */
.schedule-section {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 10px;
  padding: 16px;
}

.schedule-table-wrap { overflow-x: auto; }

.schedule-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.schedule-table th {
  text-align: left;
  padding: 6px 10px;
  color: #64748b;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  border-bottom: 1px solid #2d3148;
}

.schedule-table td {
  padding: 8px 10px;
  color: #94a3b8;
  border-bottom: 1px solid #1e2235;
}

.td-question {
  max-width: 280px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #e2e8f0;
}

.row-due td {
  background: rgba(239,68,68,0.06);
}
.row-due .td-question { color: #fca5a5; }
</style>
