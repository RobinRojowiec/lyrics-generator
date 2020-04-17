<template>
  <div class="container">
    
    <div class="row">
      <div class="offset-md-4 col-md-4">
        <br />
        <h1>{{ msg }}</h1>
        <p>{{ instruction }}</p>
        <br />
      </div>
    </div>

    <div class="row">
      <div class="col-md-3 offset-md-2">
        <b-form-select v-model="input.artist" :options="options.artist"></b-form-select>
      </div>
      <div class="col-md-3">
        <b-form-select v-model="input.genre" :options="options.genre"></b-form-select>
      </div>

      <div class="col-md-2">
        <b-button v-on:click="sendData()">Generate</b-button>
      </div>
    </div>

    <div class="row">
      <div class="col-md-6 offset-md-3" v-if="loader_message">
        <b-spinner class="m-5" label="Spinning"></b-spinner>
        <p>{{loader_message}}</p>
      </div>
      <div class="col-md-6 offset-md-3" v-if="error_message">
        <br />
        <b-alert show dismissible variant="danger">{{error_message}}</b-alert>
      </div>
      <div class="col-md-6 offset-md-3 lyrics-container">
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
        text: "",
        length: {
          chars: 0,
          words: 0
        }
      },
      input: {
        genre: -1,
        artist: -1
      },
      options: {
        genre: [{value: -1, text: "Please select a genre."}],
        artist: [{value: -1, text: "Please select an artist."}]
      }
    };
  },
  mounted() {
    this.error_message = null
    this.loader_message = "Loading options..."
    axios({ method: "GET", url: "/api/parameters" }).then(
      result => {
        this.options.artist = this.options.artist.slice(0,1);
        for (var i in result.data.artist) {
          name = result.data.artist[i];
          this.options.artist.push({ value: parseInt(i)+1 , text: name });
        }

        this.options.genre = this.options.genre.slice(0,1);
        for (var i in result.data.genre) {
          name = result.data.genre[i];
          this.options.genre.push({ value: parseInt(i)+1, text: name });
        }
      },
      error => {
        this.error_message = error.toString()
      }
    ).finally(()=>{
      this.loader_message = null
    });
  },
  methods: {
    sendData() {
      this.lyrics.text = ""
      this.error_message = null
      this.loader_message = "Generating..."
      axios({
        method: "POST",
        url:
          "/api/generate?artist=" +
          this.input.artist +
          "&genre=" +
          this.input.genre +"&insert_line_breaks=True",
        headers: { "content-type": "application/json" }
      }).then(
        result => {
          this.lyrics = result.data;
        },
        error => {
          this.error_message = error.toString()
        }
      ).finally(()=>{
        this.loader_message = null
      });
    }
  }
};
</script>

<style scoped>
.lyrics-container{
  min-height: 500px;
}
</style>