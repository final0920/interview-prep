<template>
  <div class="view-shell">
    <div class="view-header">
      <h1 class="view-title">记忆时间线</h1>
      <button class="btn btn-secondary" @click="memStore.fetch()">刷新</button>
    </div>

    <div v-if="memStore.loading" class="status-msg">加载中...</div>
    <div v-else-if="memStore.error" class="status-msg error">{{ memStore.error }}</div>

    <div v-else class="memory-layout">
      <!-- Timeline of episodes -->
      <section class="timeline-section">
        <div class="section-title">事件记录</div>
        <div v-if="!memStore.episodes.length" class="empty-msg">暂无记录</div>
        <div class="timeline">
          <div
            v-for="ep in sortedEpisodes"
            :key="ep.id ?? ep.ts"
            class="timeline-item"
          >
            <div class="tl-dot" />
            <div class="tl-content">
              <div class="tl-meta">
                <span class="tl-time">{{ formatTs(ep.ts) }}</span>
                <span v-for="tag in (ep.tags ?? [])" :key="tag" class="tl-tag">{{ tag }}</span>
              </div>
              <div class="tl-text">{{ ep.text }}</div>
              <div v-if="ep.score != null" class="tl-score">评分: {{ ep.score }}</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Semantic memory -->
      <section class="semantic-section">
        <div class="section-title">语义记忆</div>
        <div v-if="!memStore.semantic.length" class="empty-msg">暂无语义记忆</div>
        <div class="semantic-grid">
          <div
            v-for="sm in memStore.semantic"
            :key="sm.key + sm.valid_from"
            class="semantic-card"
          >
            <div class="sm-key">{{ sm.key }}</div>
            <div class="sm-kind">{{ sm.kind }}</div>
            <div class="sm-value">{{ sm.value }}</div>
          </div>
        </div>
      </section>

      <!-- Weakpoints summary -->
      <section class="weakpoints-section">
        <div class="section-title">弱点汇总</div>
        <div v-if="!memStore.weakpoints.length" class="empty-msg">暂无弱点</div>
        <ul class="weakpoints-list">
          <li v-for="wp in memStore.weakpoints" :key="wp" class="wp-item">{{ wp }}</li>
        </ul>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useMemoryStore } from '../stores/memory'

const memStore = useMemoryStore()

onMounted(() => memStore.fetch())

const sortedEpisodes = computed(() =>
  [...memStore.episodes].sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0))
)

function formatTs(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString('zh-CN')
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

.memory-layout {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  color: #7c86ff;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

.empty-msg {
  font-size: 13px;
  color: #4b5563;
}

/* Timeline */
.timeline {
  display: flex;
  flex-direction: column;
  gap: 0;
  border-left: 2px solid #2d3148;
  padding-left: 16px;
}

.timeline-item {
  display: flex;
  gap: 12px;
  position: relative;
  padding-bottom: 20px;
}

.tl-dot {
  position: absolute;
  left: -21px;
  top: 4px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #7c86ff;
  border: 2px solid #0f1117;
}

.tl-content { flex: 1; }

.tl-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
  flex-wrap: wrap;
}

.tl-time {
  font-size: 11px;
  color: #6b7280;
}

.tl-tag {
  font-size: 11px;
  background: #2d3148;
  border-radius: 4px;
  padding: 1px 6px;
  color: #94a3b8;
}

.tl-text {
  font-size: 14px;
  color: #e2e8f0;
  line-height: 1.5;
}

.tl-score {
  font-size: 12px;
  color: #4ade80;
  margin-top: 4px;
}

/* Semantic grid */
.semantic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}

.semantic-card {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 8px;
  padding: 12px;
}

.sm-key {
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
  margin-bottom: 2px;
}

.sm-kind {
  font-size: 11px;
  color: #6b7280;
  margin-bottom: 6px;
}

.sm-value {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.4;
}

/* Weakpoints list */
.weakpoints-list {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.wp-item {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 13px;
  color: #fca5a5;
}
</style>
