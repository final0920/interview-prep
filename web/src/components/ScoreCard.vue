<template>
  <div class="score-card">
    <div class="card-title">评分面板</div>

    <div v-if="!evaluation" class="no-data">暂无评分数据</div>

    <template v-else>
      <!-- Overall score ring -->
      <div class="score-ring-wrap">
        <svg class="score-ring" viewBox="0 0 80 80">
          <circle class="ring-bg" cx="40" cy="40" r="32" />
          <circle
            class="ring-fill"
            cx="40" cy="40" r="32"
            :stroke-dasharray="`${ringLength} ${ringCircumference}`"
            stroke-linecap="round"
          />
        </svg>
        <div class="ring-label">
          <span class="ring-score">{{ evaluation.score ?? '-' }}</span>
          <span class="ring-max">/10</span>
        </div>
      </div>

      <div class="verdict">{{ evaluation.verdict }}</div>

      <!-- Dimension scores -->
      <div v-if="evaluation.dimensions" class="dimensions">
        <div
          v-for="(val, key) in evaluation.dimensions"
          :key="key"
          class="dim-row"
        >
          <span class="dim-label">{{ key }}</span>
          <div class="dim-bar-wrap">
            <div class="dim-bar" :style="{ width: `${(val / 10) * 100}%` }" />
          </div>
          <span class="dim-val">{{ val }}</span>
        </div>
      </div>

      <!-- Feedback -->
      <div v-if="evaluation.feedback" class="feedback-block">
        <div class="block-title">反馈</div>
        <p>{{ evaluation.feedback }}</p>
      </div>

      <!-- Missing points -->
      <div v-if="evaluation.missing_points?.length" class="missing-block">
        <div class="block-title">缺失要点</div>
        <ul>
          <li v-for="pt in evaluation.missing_points" :key="pt">{{ pt }}</li>
        </ul>
      </div>

      <!-- Evidence citations -->
      <div v-if="evaluation.evidence_ids?.length" class="evidence-block">
        <div class="block-title">引用证据</div>
        <div class="evidence-tags">
          <span v-for="id in evaluation.evidence_ids" :key="id" class="ev-tag">{{ id }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  evaluation: { type: Object, default: null },
})

// SVG ring math
const ringCircumference = computed(() => 2 * Math.PI * 32)
const ringLength = computed(() => {
  const score = props.evaluation?.score ?? 0
  return (score / 10) * ringCircumference.value
})
</script>

<style scoped>
.score-card {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 10px;
  padding: 16px;
  flex-shrink: 0;
}

.card-title {
  font-size: 12px;
  font-weight: 700;
  color: #7c86ff;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

.no-data {
  color: #4b5563;
  font-size: 13px;
  text-align: center;
  padding: 16px 0;
}

/* Score ring */
.score-ring-wrap {
  position: relative;
  width: 80px;
  margin: 0 auto 8px;
}
.score-ring {
  width: 80px;
  height: 80px;
  transform: rotate(-90deg);
}
.ring-bg {
  fill: none;
  stroke: #2d3148;
  stroke-width: 6;
}
.ring-fill {
  fill: none;
  stroke: #4ade80;
  stroke-width: 6;
  transition: stroke-dasharray 0.4s ease;
}
.ring-label {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1px;
}
.ring-score {
  font-size: 20px;
  font-weight: 700;
  color: #4ade80;
}
.ring-max {
  font-size: 11px;
  color: #6b7280;
  margin-top: 4px;
}

.verdict {
  text-align: center;
  font-size: 13px;
  color: #86efac;
  margin-bottom: 12px;
}

/* Dimension bars */
.dimensions {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
}
.dim-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.dim-label {
  width: 72px;
  color: #94a3b8;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.dim-bar-wrap {
  flex: 1;
  height: 4px;
  background: #2d3148;
  border-radius: 2px;
  overflow: hidden;
}
.dim-bar {
  height: 100%;
  background: #7c86ff;
  border-radius: 2px;
  transition: width 0.4s ease;
}
.dim-val {
  width: 18px;
  text-align: right;
  color: #e2e8f0;
}

/* Text blocks */
.feedback-block,
.missing-block,
.evidence-block {
  border-top: 1px solid #2d3148;
  padding-top: 10px;
  margin-top: 10px;
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
}
.block-title {
  font-weight: 600;
  color: #64748b;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}
.missing-block ul {
  padding-left: 16px;
}
.evidence-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.ev-tag {
  background: #2d3148;
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 11px;
  color: #94a3b8;
}
</style>
