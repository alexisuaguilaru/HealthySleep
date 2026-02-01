<script setup>
const HOME_URL = import.meta.env.VITE_DOMAIN_URL || '"http://localhost:8080"';
const STATISTICAL_ANALYSIS_URL = import.meta.env.VITE_STATISTICAL_ANALYSIS_URL || '"http://localhost:5050"';
const DATA_MINING_URL = import.meta.env.VITE_DATA_MINING_URL || '"http://localhost:5151"';
</script>

<template>
  <nav>
    <div class="nav-links">
      <a :href="HOME_URL" target="_blank" title="Home Page">
        <i class="fa fa-home"></i> Home Page
      </a>
      |
      <a :href="STATISTICAL_ANALYSIS_URL" target="_blank" title="Statistical Analysis Notebook">
        <i class="fa fa-bar-chart"></i> Statistical Analysis
      </a>
      |
      <a :href="DATA_MINING_URL" target="_blank" title="Data Mining Notebook">
        <i class="fa fa-database"></i> Data Mining
      </a>
    </div>
  </nav>
</template>

<style scoped>
.nav-links {
  width: 100%;
  margin-top: 50px;
  padding: 20px 0;
  border-top: 1px solid #e0e0e0;
  text-align: center;
  color: #555;
  font-size: 20px;
}
a {
  color: #007bff;
  text-decoration: none;
  margin: 0 10px;
  transition: color 0.3s;
  font-weight: 600;
}
a:hover {
  color: #0056b3;
  text-decoration: underline;
}
</style>