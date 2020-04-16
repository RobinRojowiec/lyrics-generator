<template>
  <div class="container">
    <h1>{{ msg }}</h1>

    <div class="row">
      <div class="offset-md-4 col-md-4">
        <p>{{ instruction }}</p>
        <br />
      </div>
    </div>

    <div class="row">
      <div class="col-md-2 offset-md-3">
        <b-form-select v-model="input.artist" :options="options.artist"></b-form-select>
      </div>
      <div class="col-md-2">
        <b-form-select v-model="input.genre" :options="options.genre"></b-form-select>
      </div>

      <div class="col-md-2">
        <b-button v-on:click="sendData()">Generate</b-button>
      </div>
    </div>

    <div class="row">
      <div class="col-md-6 offset-md-3" v-if="loading">
        <b-spinner class="m-5" label="Spinning"></b-spinner>
        <p>Generating...</p>
      </div>
      <div class="col-md-6 offset-md-3" v-if="!loading">
        <br />
        <p v-for="(text, index) in lyrics.text.split('\n')" :key="index" v-html="text"></p>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "GeneratorForm",
  props: {
    msg: String,
    instruction: String
  },
  data() {
    return {
      loading: false,
      lyrics: {
        text: "",
        length: {
          chars: 0,
          words: 0
        }
      },
      input: {
        genre: "",
        artist: ""
      },
      options: {
        genre: [],
        artist: []
      }
    };
  },
  mounted() {
    axios({ method: "GET", url: "/api/parameters" }).then(
      result => {
        this.options.artist = [];
        for (var i in result.data.artist) {
          name = result.data.artist[i];
          this.options.artist.push({ value: parseInt(i)+1 , text: name });
        }

        this.options.genre = [];
        for (var i in result.data.genre) {
          name = result.data.genre[i];
          this.options.genre.push({ value: parseInt(i)+1, text: name });
        }
      },
      error => {
        console.error(error);
      }
    );
  },
  methods: {
    sendData() {
      this.loading = true
      axios({
        method: "POST",
        url:
          "/api/generate?artist=" +
          this.input.artist +
          "&genre=" +
          this.input.genre,
        headers: { "content-type": "application/json" }
      }).then(
        result => {
          this.lyrics = result.data;
          this.loading = false
        },
        error => {
          console.error(error);
          this.loading = false
        }
      );
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
