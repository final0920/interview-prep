<template>
  <div class="heatmap-card">
    <div class="card-title">弱点热力图</div>

    <div v-if="memStore.loading" class="status-msg">加载中...</div>
    <div v-else-if="memStore.error" class="status-msg error">{{ memStore.error }}</div>
    <div v-else-if="!memStore.weakpoints.length" class="status-msg muted">暂无弱点数据</div>

    <div v-else class="heatmap-grid">
      <div
        v-for="(wp, i) in memStore.weakpoints"
        :key="wp"
        class="heat-cell"
        :style="{ background: cellColor(i) }"
        :title="wp"
      >
        {{ wp }}
      </div>
    </div>

    <button class="refresh-btn" @click="memStore.fetch()">刷新</button>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useMemoryStore } from '../stores/memory'

const memStore = useMemoryStore()

onMounted(() => {
  if (!memStore.weakpoints.length) memStore.fetch()
})

// Gradient from high-severity red to lower-severity amber based on rank.
function cellColor(index) {
  const total = memStore.weakpoints.length
  // index 0 = worst; fade from #ef4444 -> #f59e0b -> #22d3ee
  const ratio = total <= 1 ? 0 : index / (total - 1)
  if (ratio < 0.5) {
    // red -> amber
    const t = ratio / 0.5
    const r = Math.round(239 + (245 - 239) * t)
    const g = Math.round(68  + (158 - 68)  * t)
    const b = Math.round(68  + (11  - 68)  * t)
    return `rgba(${r},${g},${b},0.25)`
  } else {
    // amber -> cyan
    const t = (ratio - 0.5) / 0.5
    const r = Math.round(245 + (34  - 245) * t)
    const g = Math.round(158 + (211 - 158) * t)
    const b = Math.round(11  + (238 - 11)  * t)
    return `rgba(${r},${g},${b},0.20)`
  }
}
</script>

<style scoped>
.heatmap-card {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 10px;
  padding: 16px;
}

.card-title {
  font-size: 12px;
  font-weight: 700;
  color: #f59e0b;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

.status-msg {
  font-size: 13px;
  color: #4b5563;
  text-align: center;
  padding: 12px 0;
}
.status-msg.error { color: #f87171; }
.status-msg.muted { color: #4b5563; }

.heatmap-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.heat-cell {
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 12px;
  color: #e2e8f0;
  border: 1px solid rgba(255,255,255,0.08);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 140px;
  cursor: default;
  transition: opacity 0.15s;
}
.heat-cell:hover { opacity: 0.75; }

.refresh-btn {
  width: 100%;
  padding: 6px;
  background: transparent;
  border: 1px solid #2d3148;
  border-radius: 6px;
  color: #64748b;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.refresh-btn:hover { background: #2d3148; }
</style>
