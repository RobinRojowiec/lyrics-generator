<template>
  <div class="pure-form">
    <h1>{{ msg }}</h1>

    <div class="pure-g">
      <div class="pure-u-1-3">
        <p></p>
      </div>
      <div class="pure-u-1-3">
        <p>{{ instruction }}</p>
        <br />
      </div>
      <div class="pure-u-1-3">
        <p></p>
      </div>
    </div>

    <div class="pure-g">
      <div class="pure-u-1-6">
      </div>
      <div class="pure-u-1-6">
      </div>
      <div class="pure-u-1-6">
        <div class="pure-u-1-2">
        <select id="artist" v-model="input.artist" class="pure-input-1">
          <option v-for="(item, index) in params.artist" :key="index" :value="index+1">{{item}}</option>
        </select>
        </div>
      </div>
      <div class="pure-u-1-6">
        <div class="pure-u-1-2">
          <select id="genre" v-model="input.genre" class="pure-input-1">
            <option v-for="(item, index) in params.genre" :key="index" :value="index+1">{{item}}</option>
          </select>
        </div>
      </div>
    </div>

    <div class="pure-g">
      <div class="pure-u-2-5"></div>
        <div class="pure-u-1-5">
        <button class="pure-button" v-on:click="sendData()">Generate</button>
      </div>
    </div>

    <div class="pure-g">
      <div class="pure-u-1-3">
        <p></p>
      </div>
      <div class="pure-u-1-3">
        <p v-html="lyrics.text"></p>
      </div>
      <div class="pure-u-1-3">
        <p></p>
      </div>
    </div>

  </div>
</template>

<script>
import axios from "axios";
export default {
  name: "HelloWorld",
  props: {
    msg: String,
    instruction: String
  },
  data() {
    return {
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
      params: {
        genre: [],
        artist: []
      }
    };
  },
  mounted() {
    axios({ method: "GET", url: "/api/parameters" }).then(
      result => {
        this.params = result.data;
      },
      error => {
        console.error(error);
      }
    );
  },
  methods: {
    sendData() {
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
        },
        error => {
          console.error(error);
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
