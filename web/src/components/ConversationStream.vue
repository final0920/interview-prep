<template>
  <div class="stream" ref="containerRef">
    <div v-if="turns.length === 0" class="empty-hint">
      点击「开始面试」进入模拟对话
    </div>
    <TransitionGroup name="bubble" tag="div" class="bubble-list">
      <ChatBubble
        v-for="turn in turns"
        :key="turn.id"
        :turn="turn"
      />
    </TransitionGroup>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import ChatBubble from './ChatBubble.vue'

const props = defineProps({
  turns: { type: Array, required: true },
})

const containerRef = ref(null)

// Auto-scroll to bottom whenever a new turn arrives.
watch(
  () => props.turns.length,
  async () => {
    await nextTick()
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  }
)
</script>

<style scoped>
.stream {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  scroll-behavior: smooth;
}

.empty-hint {
  color: #4b5563;
  font-size: 14px;
  text-align: center;
  margin-top: 60px;
}

.bubble-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.bubble-enter-active { transition: all 0.2s ease; }
.bubble-enter-from { opacity: 0; transform: translateY(8px); }
</style>
