<template>
  <div id="app">
    <img alt="Vue logo" class="pure-img img-center" src="./assets/bg.jpg">
    <generator-form msg="Lyrics Generator" instruction="Please select an artist and genre to generate some awesome lyrics."/>
  </div>
</template>

<script>
import GeneratorForm from './components/GeneratorForm.vue'

export default {
  name: 'App',
  components: {
    GeneratorForm
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
