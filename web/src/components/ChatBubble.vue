<template>
  <div :class="['bubble', `bubble--${turn.type}`]">
    <!-- Interviewer question -->
    <template v-if="turn.type === 'question'">
      <div class="bubble__label">面试官</div>
      <div class="bubble__body">{{ turn.payload?.text ?? turn.payload }}</div>
    </template>

    <!-- Follow-up probe -->
    <template v-else-if="turn.type === 'followup'">
      <div class="bubble__label bubble__label--followup">追问</div>
      <div class="bubble__body">{{ turn.payload?.text ?? turn.payload }}</div>
    </template>

    <!-- Candidate answer -->
    <template v-else-if="turn.type === 'answer'">
      <div class="bubble__label bubble__label--answer">你</div>
      <div class="bubble__body">{{ turn.payload }}</div>
    </template>

    <!-- Score / evaluation -->
    <template v-else-if="turn.type === 'score'">
      <div class="bubble__label bubble__label--score">评分</div>
      <div class="bubble__score">
        <span class="score-value">{{ turn.payload?.score ?? '-' }} / 10</span>
        <span class="score-verdict">{{ turn.payload?.verdict }}</span>
      </div>
      <div v-if="turn.payload?.feedback" class="bubble__feedback">
        {{ turn.payload.feedback }}
      </div>
      <div v-if="turn.payload?.missing_points?.length" class="bubble__missing">
        <span class="missing-label">缺失要点: </span>
        <span v-for="pt in turn.payload.missing_points" :key="pt" class="missing-tag">{{ pt }}</span>
      </div>
    </template>

    <!-- Session done -->
    <template v-else-if="turn.type === 'done'">
      <div class="bubble__label bubble__label--done">面试结束</div>
      <div class="bubble__body">本轮模拟面试已完成，请查看右侧评分面板。</div>
    </template>
  </div>
</template>

<script setup>
defineProps({
  turn: { type: Object, required: true },
})
</script>

<style scoped>
.bubble {
  max-width: 85%;
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 14px;
  line-height: 1.6;
}

/* Interviewer / followup: left-aligned */
.bubble--question,
.bubble--followup {
  align-self: flex-start;
  background: #1e2235;
  border: 1px solid #2d3148;
}

/* Candidate answer: right-aligned */
.bubble--answer {
  align-self: flex-end;
  background: #2a3060;
  border: 1px solid #3d4580;
}

/* Score bubble: centered, distinct color */
.bubble--score {
  align-self: center;
  background: #1a2e1a;
  border: 1px solid #2d5a2d;
  width: 100%;
  max-width: 100%;
}

/* Done banner */
.bubble--done {
  align-self: center;
  background: #1f1a2e;
  border: 1px solid #3d2d5a;
  width: 100%;
  max-width: 100%;
  text-align: center;
}

.bubble__label {
  font-size: 11px;
  font-weight: 600;
  color: #7c86ff;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.bubble__label--followup { color: #f59e0b; }
.bubble__label--answer   { color: #94a3b8; }
.bubble__label--score    { color: #4ade80; }
.bubble__label--done     { color: #a78bfa; }

.bubble__body { color: #e2e8f0; }

.bubble__score {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 4px;
}
.score-value {
  font-size: 20px;
  font-weight: 700;
  color: #4ade80;
}
.score-verdict {
  font-size: 13px;
  color: #86efac;
}

.bubble__feedback {
  color: #d1fae5;
  font-size: 13px;
  margin-top: 6px;
}

.bubble__missing {
  margin-top: 8px;
  font-size: 12px;
  color: #94a3b8;
}
.missing-label { font-weight: 600; }
.missing-tag {
  display: inline-block;
  background: #2d3148;
  border-radius: 4px;
  padding: 1px 6px;
  margin-left: 4px;
}
</style>
