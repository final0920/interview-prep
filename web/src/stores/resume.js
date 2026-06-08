import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getResume } from '../api/client'

export const useResumeStore = defineStore('resume', () => {
  const profile = ref(null)       // ResumeProfile
  const healthReport = ref(null)  // health_report dict
  const loading = ref(false)
  const error = ref('')

  async function fetch() {
    loading.value = true
    error.value = ''
    try {
      const data = await getResume()
      profile.value = data.profile ?? null
      healthReport.value = data.health_report ?? null
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { profile, healthReport, loading, error, fetch }
})
