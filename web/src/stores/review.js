import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getReview } from '../api/client'

export const useReviewStore = defineStore('review', () => {
  const schedule = ref([])        // SM-2 schedule items
  const qualityReport = ref(null) // quality_report dict
  const loading = ref(false)
  const error = ref('')

  async function fetch() {
    loading.value = true
    error.value = ''
    try {
      const data = await getReview()
      schedule.value = data.schedule ?? []
      qualityReport.value = data.quality_report ?? null
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { schedule, qualityReport, loading, error, fetch }
})
