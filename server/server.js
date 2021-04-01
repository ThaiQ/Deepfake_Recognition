var express = require('express');
var path = require('path')
var app = express();

//Path to img file directory
var dir = path.join(__dirname, '../data');
app.use('/img',express.static(dir));

//launch server
var server = app.listen(8080, function () {
   var port = server.address().port
   console.log("Example app listening at http://localhost:%s", port)
})