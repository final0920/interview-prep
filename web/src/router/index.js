import { createRouter, createWebHistory } from 'vue-router'
import InterviewView from '../views/InterviewView.vue'
import MemoryView from '../views/MemoryView.vue'
import ResumeView from '../views/ResumeView.vue'
import ReviewView from '../views/ReviewView.vue'

const routes = [
  { path: '/', component: InterviewView },
  { path: '/memory', component: MemoryView },
  { path: '/resume', component: ResumeView },
  { path: '/review', component: ReviewView },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
