<template>
  <div class="container">
    <div class="row">
      <div class="offset-md-3 col-md-6">
        <br />
        <h1>
          {{ msg }}
          <b-badge variant="danger">Beta</b-badge>
        </h1>
        <p>{{ instruction }}</p>
        <br />
      </div>
    </div>

    <div class="row">
      <div class="col-md-2 offset-md-2">
        <b-form-select
          v-model="input.artist"
          :options="options.artist"
          v-on:change="checkAndSubmit()"
        ></b-form-select>
      </div>
      <div class="col-md-2">
        <b-form-select
          v-model="input.genre"
          :options="options.genre"
          v-on:change="checkAndSubmit()"
        ></b-form-select>
      </div>
            <div class="col-md-2">
        <b-form-select
          v-model="input.keywords"
          :options="options.keywords"
          v-on:change="checkAndSubmit()"
        ></b-form-select>
      </div>
      <div class="col-md-2">
        <b-button v-on:click="checkAndSubmit()" class="left">
          <b-icon icon="arrow-counterclockwise" aria-hidden="true"></b-icon>
          Generate
        </b-button>
      </div>
    </div>

    <div class="row">
      <div class="col-md-8 offset-md-2" v-if="loader_message">
        <b-spinner class="m-5" label="Spinning"></b-spinner>
        <p>{{loader_message}}</p>
      </div>
      <div class="col-md-8 offset-md-2" v-if="error_message">
        <br />
        <b-alert show dismissible variant="danger">{{error_message}}</b-alert>
      </div>
      <div class="col-md-8 offset-md-2 lyrics-container">
        <br />
        <b v-if="lyrics.title">Title: {{lyrics.title}}</b>
        <br />
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
      error_message: "",
      loader_message: "",
      loading: false,
      lyrics: {
        title: "",
        text: "",
        length: {
          chars: 0,
          words: 0
        }
      },
      input: {
        genre: 0,
        artist: 0,
        keywords: 0,
        title: ""
      },
      options: {
        genre: [],
        artist: [],
        keywords: []
      }
    };
  },
  mounted() {
    this.error_message = null;
    this.loader_message = "Loading options...";
    axios({ method: "GET", url: "/api/parameters" })
      .then(
        result => {
          // set first element for artist
          var first = { value: 0, text: "Please select an artist." };
          this.options.artist = [];
          for (var i in result.data.artist) {
            name = result.data.artist[i];
            this.options.artist.push({ value: parseInt(i) + 1, text: name });
          }
          this.options.artist.sort((a, b) => {
            return a.text > b.text;
          });
          this.options.artist.unshift(first);

          // set first element for genre
          var first = { value: 0, text: "Please select a genre." };
          this.options.genre = [];
          for (var i in result.data.genre) {
            name = result.data.genre[i];
            this.options.genre.push({ value: parseInt(i) + 1, text: name });
          }
          this.options.genre.sort((a, b) => {
            return a.text > b.text;
          });
          this.options.genre.unshift(first);

          // set first element for keywords
          var first = { value: 0, text: "Keyword" };
          this.options.keywords = [];
          for (var i in result.data.keywords) {
            name = result.data.keywords[i];
            this.options.keywords.push({ value: parseInt(i) + 1, text: name });
          }
          this.options.keywords.sort((a, b) => {
            return a.text > b.text;
          });
          this.options.keywords.unshift(first);
        },
        error => {
          this.error_message = error.toString();
        }
      )
      .finally(() => {
        this.loader_message = null;
      });
  },
  methods: {
    checkAndSubmit() {
      if (this.input.genre > 0 && this.input.artist > 0) {
        this.sendData();
      }
    },
    sendData() {
      this.lyrics.text = "";
      this.lyrics.title = "";
      this.error_message = null;
      this.loader_message = "Generating...";
      axios({
        method: "POST",
        url:
          "/api/generate?artist=" +
          this.input.artist +
          "&genre=" +
          this.input.genre + "&keyword=" 
          + this.input.keywords,
        headers: { "content-type": "application/json" }
      })
        .then(
          result => {
            this.lyrics = result.data;
          },
          error => {
            this.error_message = error.toString();
          }
        )
        .finally(() => {
          this.loader_message = null;
        });
    }
  }
};
</script>

<style scoped>
.lyrics-container {
  min-height: 500px;
  text-align: left;
}
.btn.left{
  float: left;
  width: 100%;
}
.btn, select{
  margin-top: 10px;
}
</style>