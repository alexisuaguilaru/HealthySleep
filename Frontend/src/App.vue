<script setup>
    import PatientForm from './components/PatientForm.vue'
</script>

<template>
    <div id="app-wrapper">
        <header><h1>How Well Do You Sleep?</h1></header> 
    
    <main> 
        <PatientForm />
    </main>
    
    <footer>
        &copy; 2025 Alexis Aguilar. <a href="https://github.com/alexisuaguilaru/HealthySleep" target="_blank">GitHub</a>
    </footer>
  </div>
</template>

<style scoped>
#app-wrapper {
  display: flex;
  flex-direction: column;
  min-height: 100vh; 
  font-family: Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #412f4f;
}

header {
  background-color: #B57EDC;
  color: black;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h1 {
  margin: 0;
}

main {
  flex-grow: 1; 
  padding: 20px;
  background-color: #C9AFDB;
}

footer {
  background-color: #E1D02A;
  color: black;
  padding: 1rem;
  margin-top: auto; 
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
}

footer a {
  color: #7f20a8;
  text-decoration: none;
}

footer a:hover {
  text-decoration: underline;
}
</style>