<template>
  <div id="app">
    <img alt="Vue logo" class="pure-img img-center" src="./assets/bg.jpg">
    <HelloWorld msg="Lyrics Generator" instruction="Please select an artist and genre to generate some awesome lyrics."/>
  </div>
</template>

<script>
import HelloWorld from './components/HelloWorld.vue'

export default {
  name: 'App',
  components: {
    HelloWorld
  }
}
</script>

<style>
@import 'assets/pure-min.css';
@import 'assets/grids-responsive-min.css';
#app {
  text-align: center;
}
.img-center{
  margin-left: auto;
  margin-right: auto;
}

</style>
