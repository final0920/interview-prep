<template>
  <div class="app-shell">
    <nav class="sidebar-nav">
      <div class="logo">Interview Coach</div>
      <RouterLink to="/" class="nav-item">模拟面试</RouterLink>
      <RouterLink to="/memory" class="nav-item">记忆时间线</RouterLink>
      <RouterLink to="/resume" class="nav-item">简历分析</RouterLink>
      <RouterLink to="/review" class="nav-item">复习计划</RouterLink>
    </nav>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.sidebar-nav {
  width: 200px;
  flex-shrink: 0;
  background: #1a1d27;
  border-right: 1px solid #2d3148;
  display: flex;
  flex-direction: column;
  padding: 16px 0;
  gap: 4px;
}

.logo {
  font-size: 14px;
  font-weight: 700;
  color: #7c86ff;
  padding: 0 16px 16px;
  border-bottom: 1px solid #2d3148;
  margin-bottom: 8px;
}

.nav-item {
  display: block;
  padding: 10px 16px;
  color: #94a3b8;
  text-decoration: none;
  font-size: 14px;
  border-radius: 6px;
  margin: 0 8px;
  transition: background 0.15s, color 0.15s;
}

.nav-item:hover,
.nav-item.router-link-active {
  background: #2d3148;
  color: #e2e8f0;
}

.content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
