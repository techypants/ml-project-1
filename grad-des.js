var data = [];

var m = 0;
var b = 0;

var gradientInfoDiv;

function setup() {
  const cnv =createCanvas(400, 400); 
  cnv.id('cnv');
  cnv.parent('cont');
  gradientInfoDiv = createDiv('');
  gradientInfoDiv.id('gradientInfo');
  gradientInfoDiv.parent('cont');
}

function gradientDescent() {
  var learning_rate = 0.05;
  for (var i = 0; i < data.length; i++) {
    var x = data[i].x;
    var y = data[i].y;
    var guess = m * x + b;
    var error = y - guess;
    m = m + error * x * learning_rate;
    b = b + error * learning_rate;
  }

  gradientInfoDiv.html('Gradient Descent Info:<br>' +
  'Slope (m): ' + m.toFixed(2) + '<br>' +
  'Intercept (b): ' + b.toFixed(2));
}

function drawLine() {
  var x1 = 0;
  var y1 = m * x1 + b;
  var x2 = 1;
  var y2 = m * x2 + b;

  x1 = map(x1, 0, 1, 0, width);
  y1 = map(y1, 0, 1, height, 0);
  x2 = map(x2, 0, 1, 0, width);
  y2 = map(y2, 0, 1, height, 0);

  stroke(255);
  strokeWeight(2);
  line(x1, y1, x2, y2);
}

function mousePressed() {
  
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    var x = map(mouseX, 0, width, 0, 1);
    var y = map(mouseY, 0, height, 1, 0);
    var point = createVector(x, y);
    data.push(point);
  }
}


function draw() {
  background(51);
  for (var i = 0; i < data.length; i++) {
    var x = map(data[i].x, 0, 1, 0, width);
    var y = map(data[i].y, 0, 1, height, 0);
    fill(255);
    stroke(255);
    ellipse(x, y, 8, 8);
  }

  if (data.length > 1) {
    gradientDescent();
    drawLine();
  }
}