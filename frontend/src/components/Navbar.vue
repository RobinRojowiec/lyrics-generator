<template>

<div>
  <b-navbar toggleable="lg" type="dark" variant="dark">
    <b-navbar-brand href="#">{{title}}</b-navbar-brand>
    <b-badge variant="danger">Beta</b-badge>
    <b-navbar-toggle target="nav-collapse"></b-navbar-toggle>
  </b-navbar>
</div>

</template>

<script>
export default {
  name: "Navbar",
  props: {
    title: String
  },
  data() {
    return {};
  },
  mounted() {
  },
  methods: {
  }
};
</script>
