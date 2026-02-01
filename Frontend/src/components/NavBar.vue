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
nav {
  color: black;
  padding: 0.5rem;
}

.nav-links {
  width: 100%;
  padding: 10px 0;
  text-align: center;
  font-size: 26px;
  color: white;
}

a {
  color: white;
  text-decoration: none;
  transition: color 0.3s;
  font-weight: 600;
}

a:hover {
  color: #7f20a8;
  text-decoration: underline;
}
</style>