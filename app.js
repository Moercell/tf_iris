const tf = require('@tensorflow/tfjs-node'); // import node binding (optional ohne node)
const data = require('./data.json');
const testing = require('./testing.json');

//* prepare Data
const trainingData = tf.tensor2d(data.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]));
const outputData = tf.tensor2d(data.map(item => [
    item.species === "setosa" ? 1 : 0,
    item.species === "virginica" ? 1 : 0,
    item.species === "versicolor" ? 1 : 0,
]));

const testingData = tf.tensor2d(testing.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
]));


const model = tf.sequential(); //! init network

//* add layers (configure network)
model.add(tf.layers.dense({
    inputShape: [4], // because 4 values
    activation: "sigmoid", // good for 1 / 0 values (classify) //? see: link01
    units: 5, // next network "row"
}));

model.add(tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3,
}));
model.add(tf.layers.dense({
    // no input
    activation: "sigmoid",
    units: 3,
}));

model.compile({
    loss: "meanSquaredError", //? see link02
    optimizer: tf.train.adam(.06), //? see link03
});

//! MAGIC
model.fit(trainingData, outputData, { epochs: 100 }) 
    .then((history) => {
        //console.log(history) //* Genauigkeit 
        model.predict(testingData).print() //* testing
    });



//? link list
// link01: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
// link02: https://www.tensorflow.org/api_docs/python/tf/losses
// link03: https://www.tensorflow.org/api_docs/python/tf/train