<template>

  <footer class="page-footer font-small blue">

    <!-- Copyright -->
    <div class="footer-copyright text-center">Robin Rojowiec 2020. 
      <a href="https://www.rojowiec.de">www.rojowiec.de</a>
    </div>
    <!-- Copyright -->

  </footer>

</template>

<script>
export default {
  name: "Footer",
  props: {}
};
</script>

<style scoped>
.page-footer{
  display: flex;
  width: 100%;
  background: #343A40;
}
.footer-copyright{
  padding: 10px;
  color: white
}
.footer-copyright a{
  color: lightblue;
  text-decoration: none
}
.text-center{
  text-align: center;
}
</style>
