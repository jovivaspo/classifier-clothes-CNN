import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js'

var interval = 2000;

const RANGER = document.getElementById('ranger');

const DOM_SPEED = document.getElementById('domSpeed');


// When user drags slider update interval.

RANGER.addEventListener('input', function(e) {

  interval = this.value;

  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';

});

const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;


// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;


// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);

//Función típica de normalización para el tensor de entrada

function normalize(tensor, min, max) {

    const result = tf.tidy(function() {
  
      const MIN_VALUES = tf.scalar(min);
  
      const MAX_VALUES = tf.scalar(max);
  
  
      const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
  
      const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
  
      const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
  
  
      return NORMALIZED_VALUES;
  
    });
  
    return result;
  
  }
  
  
  // Input feature Array is 2 dimensional.
  
  const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);


// Output feature Array is 1 dimensional.

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define model architecture.

const model = tf.sequential();

/*Creación de capas:
1. En primer lugar se define una capa convolucional2D con forma [28,28,1]
Tendremos 16 filtros de tamaño 3*3, stride de 1(el valor del filtro se calcula para cada pixel de la entrada)
padding= 'same' para asegurarnos de que entran todos los pixeles, y por ultimo funcion relu

*/
model.add(tf.layers.conv2d({

    inputShape: [28, 28, 1],
  
    filters: 16,
  
    kernelSize: 3, // Square Filter of 3 by 3. Could also specify rectangle eg [2, 3].
  
    strides: 1,
  
    padding: 'same',
  
    activation: 'relu'  
  
  }));
  /*
  2.Como segunda capa tenemos una max pol2d de tamaño 2 y stride 2
  */
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  /*
  3. Se añade otra capa convolucional con el doble de filtros que la segunda
  */
 
model.add(tf.layers.conv2d({

    filters: 32,
  
    kernelSize: 3,
  
    strides: 1,
  
    padding: 'same',
  
    activation: 'relu'
  
  }));
  /* 4.De nuevo añadimos una max pool que reducirá el tamaño a la mitad*/
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  /*
  A continuación, tendremos una capa densa de 128 neuronas, pero antes se debe convertir la salida
  a tipo lista, ya que tenemos 32 salidas de 7*7, finamente tendremos la capa final de salida
  */ 

  
model.add(tf.layers.flatten());


model.add(tf.layers.dense({units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();


train();

async function train() { 

    // Compile the model with the defined optimizer and specify our loss function to use.
  
    model.compile({
  
      optimizer: 'adam',
  
      loss: 'categoricalCrossentropy',
  
      metrics: ['accuracy'] //measure of how many images are predicted correctly from the training data.
  
    });

    function logProgress(epoch, logs) {

   
      console.log('Data for epoch ' + epoch, logs);

    }
  
    const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]); //longitud, tamañao de las imágenes, y 1 porque son greyscale
    let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
  
      shuffle: true,        // Ensure data is shuffled again before using each epoch.
  
      validationSplit: 0.15,
  
      batchSize: 256,       // Update weights after every 256 examples.      
  
      epochs: 30,           // Go over the data 50 times!

      callbacks: {onEpochEnd: logProgress},
  
  
    });
  
    
    RESHAPED_INPUTS.dispose();

    OUTPUTS_TENSOR.dispose();
  
    INPUTS_TENSOR.dispose();

    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  
    evaluate(); // Once trained we can evaluate the model.
  
  }

  const PREDICTION_ELEMENT = document.getElementById('prediction');


function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs. 


  let answer = tf.tidy(function() {

    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]),0,255)

    let output = model.predict(newInput.reshape([1,28,28,1]));

    output.print();

    return output.squeeze().argMax();    

  });

  answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = LOOKUP[index];

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}

const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28); //Origen  0,0 y tamaño imagen

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }


  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 


  // Perform a new classification after a certain interval.

  setTimeout(evaluate, interval);

}