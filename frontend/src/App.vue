<template>
  <div id="app">
    <Navbar :title="title" />
    <img :alt="title" class="img-center" src="./assets/bg.jpg" />
    <GeneratorForm :msg="title" :instruction="description" />
    <Footer />
  </div>
</template>

<script>
import GeneratorForm from './components/GeneratorForm.vue'
import Navbar from './components/Navbar.vue'
import Footer from './components/Footer.vue'

export default {
  name: 'App',
  components: {
    GeneratorForm,
    Navbar,
    Footer
  },
  data: function(){
    return {
      title: "Lyrigen",
      description: "Please select an artist and genre to generate some awesome lyrics."
    }
  }
}
</script>

<style>
body, html{
  min-height: 100%;
  padding: 0;
  margin: 0;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  min-height: 100%;
}
</style>
