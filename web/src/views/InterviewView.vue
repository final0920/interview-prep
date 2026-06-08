<template>
  <div class="interview-layout">
    <!-- Left: conversation stream -->
    <section class="chat-panel">
      <div class="chat-header">
        <span class="role-label">目标岗位: {{ store.targetRole }}</span>
        <div class="header-actions">
          <input
            v-if="!store.isActive && !store.isDone"
            v-model="roleInput"
            class="role-input"
            placeholder="岗位名称..."
          />
          <button
            v-if="!store.isActive && !store.isDone"
            class="btn btn-primary"
            @click="handleStart"
          >
            开始面试
          </button>
          <button
            v-if="store.isActive || store.isDone"
            class="btn btn-secondary"
            @click="store.resetSession"
          >
            重置
          </button>
        </div>
      </div>

      <ConversationStream :turns="store.turns" />

      <div v-if="store.status === 'error'" class="error-bar">{{ store.errorMsg }}</div>

      <AnswerInput
        v-if="store.isActive"
        :disabled="waitingForNext"
        @submit="handleAnswer"
      />
    </section>

    <!-- Right: score card + weakpoint heatmap -->
    <aside class="side-panel">
      <ScoreCard :evaluation="store.currentScore" />
      <WeakpointHeatmap />
    </aside>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useInterviewStore } from '../stores/interview'
import ConversationStream from '../components/ConversationStream.vue'
import AnswerInput from '../components/AnswerInput.vue'
import ScoreCard from '../components/ScoreCard.vue'
import WeakpointHeatmap from '../components/WeakpointHeatmap.vue'

const store = useInterviewStore()
const roleInput = ref('')

// Disable input while we wait for the server to respond with the next turn.
const waitingForNext = computed(() => {
  const last = store.turns[store.turns.length - 1]
  return last?.type === 'answer'
})

function handleStart() {
  store.start(roleInput.value.trim() || undefined)
}

function handleAnswer(text) {
  store.answer(text)
}
</script>

<style scoped>
.interview-layout {
  display: flex;
  height: 100%;
  overflow: hidden;
}

.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid #2d3148;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #2d3148;
  background: #1a1d27;
  flex-shrink: 0;
}

.role-label {
  font-size: 13px;
  color: #94a3b8;
}

.header-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.role-input {
  background: #0f1117;
  border: 1px solid #2d3148;
  border-radius: 6px;
  padding: 6px 10px;
  color: #e2e8f0;
  font-size: 13px;
  outline: none;
  width: 200px;
}

.btn {
  padding: 7px 14px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; }
.btn-primary { background: #7c86ff; color: #fff; }
.btn-secondary { background: #2d3148; color: #94a3b8; }

.error-bar {
  padding: 8px 16px;
  background: #3b1a1a;
  color: #f87171;
  font-size: 13px;
  flex-shrink: 0;
}

.side-panel {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  background: #13151f;
  padding: 16px;
  gap: 16px;
}
</style>
