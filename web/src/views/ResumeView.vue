<template>
  <div class="view-shell">
    <div class="view-header">
      <h1 class="view-title">简历分析</h1>
      <button class="btn btn-secondary" @click="resumeStore.fetch()">刷新</button>
    </div>

    <div v-if="resumeStore.loading" class="status-msg">加载中...</div>
    <div v-else-if="resumeStore.error" class="status-msg error">{{ resumeStore.error }}</div>

    <div v-else class="resume-layout">
      <!-- Left: profile -->
      <section class="profile-panel">
        <div class="section-title">个人档案</div>
        <div v-if="!resumeStore.profile" class="empty-msg">暂无简历数据</div>
        <template v-else>
          <div class="profile-field">
            <span class="field-label">姓名</span>
            <span class="field-val">{{ resumeStore.profile.name ?? '-' }}</span>
          </div>
          <div class="profile-field">
            <span class="field-label">目标岗位</span>
            <span class="field-val">{{ resumeStore.profile.target_role ?? '-' }}</span>
          </div>
          <div class="profile-field">
            <span class="field-label">工作年限</span>
            <span class="field-val">{{ resumeStore.profile.years_of_experience ?? '-' }} 年</span>
          </div>

          <!-- Skills -->
          <div v-if="resumeStore.profile.skills?.length" class="subsection">
            <div class="subsection-title">技能</div>
            <div class="tag-cloud">
              <span v-for="sk in resumeStore.profile.skills" :key="sk" class="skill-tag">{{ sk }}</span>
            </div>
          </div>

          <!-- Experience bullets -->
          <div v-if="resumeStore.profile.experience?.length" class="subsection">
            <div class="subsection-title">工作经历</div>
            <div v-for="exp in resumeStore.profile.experience" :key="exp.company + exp.title" class="exp-item">
              <div class="exp-header">
                <span class="exp-title">{{ exp.title }}</span>
                <span class="exp-company"> @ {{ exp.company }}</span>
              </div>
              <div class="exp-period">{{ exp.period }}</div>
              <ul v-if="exp.bullets?.length" class="exp-bullets">
                <li v-for="b in exp.bullets" :key="b">{{ b }}</li>
              </ul>
            </div>
          </div>
        </template>
      </section>

      <!-- Right: health report -->
      <section class="health-panel">
        <div class="section-title">健康报告</div>
        <div v-if="!resumeStore.healthReport" class="empty-msg">暂无分析数据</div>
        <template v-else>
          <!-- Match score -->
          <div class="metric-row">
            <span class="metric-label">岗位匹配度</span>
            <div class="metric-bar-wrap">
              <div
                class="metric-bar"
                :style="{ width: `${(resumeStore.healthReport.match_score ?? 0) * 100}%` }"
              />
            </div>
            <span class="metric-val">
              {{ pct(resumeStore.healthReport.match_score) }}
            </span>
          </div>

          <!-- ATS score -->
          <div class="metric-row">
            <span class="metric-label">ATS 得分</span>
            <div class="metric-bar-wrap">
              <div
                class="metric-bar metric-bar--ats"
                :style="{ width: `${(resumeStore.healthReport.ats_score ?? 0) * 100}%` }"
              />
            </div>
            <span class="metric-val">
              {{ pct(resumeStore.healthReport.ats_score) }}
            </span>
          </div>

          <!-- Coverage heatmap -->
          <div v-if="resumeStore.healthReport.coverage" class="coverage-section">
            <div class="subsection-title">技能覆盖</div>
            <div class="coverage-grid">
              <div
                v-for="(score, skill) in resumeStore.healthReport.coverage"
                :key="skill"
                class="cov-cell"
                :class="coverageClass(score)"
              >
                {{ skill }}
              </div>
            </div>
          </div>

          <!-- Skill gaps -->
          <div v-if="resumeStore.healthReport.gaps?.length" class="gaps-section">
            <div class="subsection-title">技能缺口</div>
            <div
              v-for="gap in resumeStore.healthReport.gaps"
              :key="gap.skill"
              class="gap-item"
              :class="`gap-item--${gap.severity ?? 'medium'}`"
            >
              <span class="gap-skill">{{ gap.skill }}</span>
              <span class="gap-sev">{{ gap.severity }}</span>
            </div>
          </div>
        </template>
      </section>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useResumeStore } from '../stores/resume'

const resumeStore = useResumeStore()

onMounted(() => resumeStore.fetch())

function pct(val) {
  if (val == null) return '-'
  return `${Math.round(val * 100)}%`
}

function coverageClass(score) {
  if (score >= 0.7) return 'cov-high'
  if (score >= 0.4) return 'cov-mid'
  return 'cov-low'
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

.resume-layout {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  align-items: start;
}

.profile-panel,
.health-panel {
  background: #1a1d27;
  border: 1px solid #2d3148;
  border-radius: 10px;
  padding: 16px;
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  color: #7c86ff;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 14px;
}

.empty-msg {
  font-size: 13px;
  color: #4b5563;
}

.profile-field {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 13px;
}
.field-label {
  color: #6b7280;
  width: 72px;
  flex-shrink: 0;
}
.field-val { color: #e2e8f0; }

.subsection { margin-top: 14px; }
.subsection-title {
  font-size: 11px;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.tag-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
.skill-tag {
  background: #2d3148;
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 12px;
  color: #94a3b8;
}

.exp-item { margin-bottom: 12px; }
.exp-header { font-size: 13px; color: #e2e8f0; }
.exp-title { font-weight: 600; }
.exp-company { color: #7c86ff; }
.exp-period { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
.exp-bullets {
  padding-left: 16px;
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
}

/* Metrics */
.metric-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 13px;
}
.metric-label { width: 80px; color: #94a3b8; flex-shrink: 0; }
.metric-bar-wrap {
  flex: 1;
  height: 6px;
  background: #2d3148;
  border-radius: 3px;
  overflow: hidden;
}
.metric-bar {
  height: 100%;
  background: #7c86ff;
  border-radius: 3px;
  transition: width 0.4s ease;
}
.metric-bar--ats { background: #4ade80; }
.metric-val { width: 36px; text-align: right; color: #e2e8f0; font-size: 12px; }

/* Coverage */
.coverage-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.cov-cell {
  border-radius: 5px;
  padding: 4px 10px;
  font-size: 12px;
  border: 1px solid transparent;
}
.cov-high { background: rgba(74,222,128,0.15); border-color: rgba(74,222,128,0.3); color: #86efac; }
.cov-mid  { background: rgba(245,158,11,0.15); border-color: rgba(245,158,11,0.3); color: #fcd34d; }
.cov-low  { background: rgba(239,68,68,0.15);  border-color: rgba(239,68,68,0.3);  color: #fca5a5; }

/* Gaps */
.gaps-section { margin-top: 12px; }
.gap-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  border-radius: 6px;
  margin-bottom: 6px;
  font-size: 13px;
  border: 1px solid transparent;
}
.gap-item--high   { background: rgba(239,68,68,0.12);  border-color: rgba(239,68,68,0.25);  color: #fca5a5; }
.gap-item--medium { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.25); color: #fcd34d; }
.gap-item--low    { background: rgba(74,222,128,0.10); border-color: rgba(74,222,128,0.20); color: #86efac; }
.gap-skill { font-weight: 500; }
.gap-sev { font-size: 11px; opacity: 0.7; }
</style>
