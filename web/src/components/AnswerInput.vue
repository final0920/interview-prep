<template>
  <div class="answer-input-bar">
    <textarea
      ref="textareaRef"
      v-model="text"
      class="answer-textarea"
      :disabled="disabled"
      placeholder="输入你的回答... (Ctrl+Enter 提交)"
      rows="3"
      @keydown.ctrl.enter.prevent="submit"
    />
    <button
      class="submit-btn"
      :disabled="disabled || !text.trim()"
      @click="submit"
    >
      提交
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  disabled: { type: Boolean, default: false },
})
const emit = defineEmits(['submit'])

const text = ref('')
const textareaRef = ref(null)

function submit() {
  const val = text.value.trim()
  if (!val || props.disabled) return
  emit('submit', val)
  text.value = ''
}
</script>

<style scoped>
.answer-input-bar {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid #2d3148;
  background: #1a1d27;
  flex-shrink: 0;
  align-items: flex-end;
}

.answer-textarea {
  flex: 1;
  background: #0f1117;
  border: 1px solid #2d3148;
  border-radius: 8px;
  padding: 10px 12px;
  color: #e2e8f0;
  font-size: 14px;
  resize: none;
  outline: none;
  line-height: 1.5;
  transition: border-color 0.15s;
}
.answer-textarea:focus { border-color: #7c86ff; }
.answer-textarea:disabled { opacity: 0.4; cursor: not-allowed; }

.submit-btn {
  padding: 10px 20px;
  background: #7c86ff;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  white-space: nowrap;
  transition: opacity 0.15s;
  align-self: flex-end;
}
.submit-btn:hover:not(:disabled) { opacity: 0.85; }
.submit-btn:disabled { opacity: 0.35; cursor: not-allowed; }
</style>
