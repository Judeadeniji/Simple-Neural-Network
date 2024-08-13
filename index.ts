import { Activation } from './activation-functions';
import sentimentModel from './model.json'
import { NeuralNetwork } from './nn'
import readline from 'node:readline'
import { wordVecs as wordVectors } from './w2v';

// Function to convert text to a vector using word2vec-like vectors
function textToVector(
  text: string,
  wordVectors: Record<string, number[]>
): number[][] {
  const words = text.toLowerCase().split(/\W+/);
  const vectors = words
    .map((word) => wordVectors[word])
    .filter((vector) => vector !== undefined);

  if (vectors.length === 0) {
    return Array(Object.values(wordVectors)[0].length).fill(0); // Return a zero vector if no known words are found
  }

  const summedVector = vectors.reduce(
    (sum, vec) => sum.map((v, i) => v + vec[i]),
    Array(vectors[0].length).fill(0) as number[]
  );
  return summedVector.map((v) => [v / vectors.length]); // Return the average vector
}


const nn = new NeuralNetwork(sentimentModel.layers, {
  activationFunction: Activation.sigmoid,
  activationFunctionDerivative: Activation.sigmoidDerivative
});

nn.load(JSON.stringify(sentimentModel));

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function start() {
  rl.question("Enter a sentence: ", (sentence) => {
    const vector = textToVector(sentence, wordVectors);
    const prediction = nn.predict(vector).flat();
    console.log("Prediction:", prediction);
    console.log("Class:", prediction.indexOf(Math.max(...prediction)) - 1);
    console.log("Negative:", prediction[0]);
    console.log("Neutral:", prediction[1]);
    console.log("Positive:", prediction[2]);
    // rl.close();
    start();
  });
}


start();