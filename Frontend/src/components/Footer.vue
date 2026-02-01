<script setup>
const CurrentYear = new Date().getFullYear();
</script>

<template>
	<footer>
    &copy; {{ CurrentYear }} | <a href="https://www.linkedin.com/in/alexis-aguilar-uribe/" target="_blank">Alexis Aguilar</a> | <a href="https://github.com/alexisuaguilaru/HealthySleep" target="_blank">GitHub Repository</a>
  </footer>
</template>

<style scoped>
footer {
  background-color: #E1D02A;
  color: black;
  padding: 1rem;
  margin-top: auto; 
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
}

footer a {
	color: black;
  text-decoration: none;
	transition: color 0.3s;
}

footer a:hover {
	color: #7f20a8;
  text-decoration: underline;
}
</style>