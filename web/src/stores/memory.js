import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getMemory } from '../api/client'

export const useMemoryStore = defineStore('memory', () => {
  const episodes = ref([])   // MemoryEpisode[]
  const semantic = ref([])   // MemorySemantic[]
  const weakpoints = ref([]) // string[]
  const loading = ref(false)
  const error = ref('')

  async function fetch() {
    loading.value = true
    error.value = ''
    try {
      const data = await getMemory()
      episodes.value = data.episodes ?? []
      semantic.value = data.semantic ?? []
      weakpoints.value = data.weakpoints ?? []
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { episodes, semantic, weakpoints, loading, error, fetch }
})
